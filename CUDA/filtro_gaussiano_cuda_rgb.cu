#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./headers/stb_image.h"
#include "./headers/stb_image_write.h"
#include <cuda_runtime.h>
#include <omp.h>

// Struct representando a imagem como double
typedef struct {
    double *R;
    double *G;
    double *B;
    int w;
    int h;
    int c;
} image_double;

// Struct representando a imagem como unsigned char
typedef struct {
    unsigned char *R;
    unsigned char *G;
    unsigned char *B;
    int w;
    int h;
    int c;
} image_char;

typedef struct {
    double *kernel_values;
    int side;
} kernel;

kernel create_gaussian_kernel(int size, double sigma); // Criar Kernel
__global__ void convolution_kernel(double *in_R, double *in_G, double *in_B, double *out_R, double *out_G, double *out_B, double *kR, double *kG, double *kB, int kside, int w, int h);
void free_matrix(double* mat, int h); // Liberar Matriz
void print_matrix(double* mat, int h, int w); // Printar Matriz
image_double open_image(char* nome); // Abrir Imagem
image_char convert_from_double(image_double); // Converter de double para unsigned char
void free_image_double(image_double img); // Liberar Imagem
void free_image_char(image_char img); // Liberar Imagem
void free_kernel(kernel Kernel); // Liberar Kernel
void save_image(char* nome, image_double blurred); //Salvar imagem



int main(void) {

    char* input_path = "./images/image_2048.png"; // Arquivo de entrada
    char* output_path = "./images/outputcuda.png"; // Arquivo de saída

    double start, stop;

    start = omp_get_wtime();
    // Abrir a imagem de entrada.
    image_double img = open_image(input_path);
    int w = img.w, h = img.h;
    stop = omp_get_wtime();
    printf("Tempo de abertura da imagem:: %f\n", stop-start);

    start = omp_get_wtime();
    // Cria kernels na CPU
    kernel kR = create_gaussian_kernel(5, 1.0);
    kernel kG = create_gaussian_kernel(5, 1.0);
    kernel kB = create_gaussian_kernel(5, 1.0);

    // Aloca memória na GPU
    double *d_R, *d_G, *d_B;           // imagem atual
    double *d_outR, *d_outG, *d_outB;  // buffer de saída
    double *d_kR, *d_kG, *d_kB;        // kernels
    size_t img_size = w * h * sizeof(double);
    size_t k_size   = 5 * 5 * sizeof(double);

    cudaMalloc(&d_R,    img_size);
    cudaMalloc(&d_G,    img_size);
    cudaMalloc(&d_B,    img_size);
    cudaMalloc(&d_outR, img_size);
    cudaMalloc(&d_outG, img_size);
    cudaMalloc(&d_outB, img_size);
    cudaMalloc(&d_kR,   k_size);
    cudaMalloc(&d_kG,   k_size);
    cudaMalloc(&d_kB,   k_size);

    // 4. Copia imagem e kernels da CPU para GPU
    cudaMemcpy(d_R,  img.R, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G,  img.G, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,  img.B, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kR, kR.kernel_values, k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kG, kG.kernel_values, k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kB, kB.kernel_values, k_size, cudaMemcpyHostToDevice);

    // 5. Configura grid e block  ← AQUI É O PASSO 2
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    // 6. Loop iterativo
    for (int it = 0; it < 100; it++) {

        convolution_kernel<<<grid, block>>>(
            d_R, d_G, d_B,
            d_outR, d_outG, d_outB,
            d_kR, d_kG, d_kB,
            5, w, h
        );

        // troca ponteiros (igual ao serial)
        double *tmp;
        tmp = d_R;    d_R    = d_outR; d_outR = tmp;
        tmp = d_G;    d_G    = d_outG; d_outG = tmp;
        tmp = d_B;    d_B    = d_outB; d_outB = tmp;
    }

    // 7. Copia resultado de volta pra CPU
    cudaDeviceSynchronize();
    cudaMemcpy(img.R, d_R, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img.G, d_G, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img.B, d_B, img_size, cudaMemcpyDeviceToHost);

    stop = omp_get_wtime();
    printf("Tempo de processamento do filtro Gaussiano: %f\n", stop-start);

    start = omp_get_wtime();
    save_image(output_path, img);
    stop = omp_get_wtime();
    printf("Tempo de salvamento da imagem:: %f\n", stop-start);

    // 6. liberar memória
    cudaFree(d_R);  
    cudaFree(d_G);  
    cudaFree(d_B);
    cudaFree(d_outR); 
    cudaFree(d_outG); 
    cudaFree(d_outB);
    cudaFree(d_kR); 
    cudaFree(d_kG); 
    cudaFree(d_kB);
    free_image_double(img);
    free_kernel(kR); 
    free_kernel(kG); 
    free_kernel(kB);
    return 0;
}


kernel create_gaussian_kernel(int size, double sigma) {
    kernel k;
    k.side = size;
    int half = size / 2;
    double twoSigmaSqr = 2.0 * sigma * sigma;

    // alocar matriz 2D
    k.kernel_values = (double*) malloc(size * size * sizeof(double));

    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        int y = i - half;

        for (int j = 0; j < size; j++) {
            int x = j - half;

            double value = exp(-(x*x + y*y) / twoSigmaSqr);

            k.kernel_values[size * i + j] = value;
            sum += value;
        }
    }

    // normalizar (soma = 1)
    double invSum = 1.0 / sum; // evita várias divisões

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            k.kernel_values[size * i + j] *= invSum;
        }
    }

    return k;
}

__global__ void convolution_kernel(double *in_R, double *in_G, double *in_B, double *out_R, double *out_G, double *out_B, double *kR, double *kG, double *kB, int kside, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // thread fora da imagem não faz nada
    if (col >= w || row >= h) return;

    int pad = kside / 2;
    double sumR = 0.0, sumG = 0.0, sumB = 0.0;

    for (int ki = 0; ki < kside; ki++) {
        for (int kj = 0; kj < kside; kj++) {

            int ri = row + ki - pad;
            int rj = col + kj - pad;

            // clamp (replicação de borda, igual ao serial)
            if (ri < 0)  ri = 0;
            if (ri >= h) ri = h - 1;
            if (rj < 0)  rj = 0;
            if (rj >= w) rj = w - 1;

            double kr = kR[ki * kside + kj];
            double kg = kG[ki * kside + kj];
            double kb = kB[ki * kside + kj];

            double pixel = in_R[ri * w + rj];
            sumR += pixel * kr;
            sumG += in_G[ri * w + rj] * kg;
            sumB += in_B[ri * w + rj] * kb;
        }
    }

    out_R[row * w + col] = sumR;
    out_G[row * w + col] = sumG;
    out_B[row * w + col] = sumB;
}


void save_image(char* output_path, image_double blurred){
    // 3. Converter de double para unsigned char pra saída.
    image_char out = convert_from_double(blurred);
    
    // 4. Montar buffer linear (RGB)
    unsigned char* buffer = (unsigned char*) malloc(out.w * out.h * 3);

    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            int idx = (i * out.w + j) * 3;

            buffer[idx]     = out.R[out.w * i + j];
            buffer[idx + 1] = out.G[out.w * i + j];
            buffer[idx + 2] = out.B[out.w * i + j];
        }
    }

    // 5. salvar imagem
    stbi_write_png(output_path, out.w, out.h, 3, buffer, out.w * 3);
    printf("Imagem salva em: %s\n", output_path);

    //Liberar memória
    free_image_char(out);
    free(buffer);
}

// libera matriz
void free_matrix(double* mat, int h) {
    free(mat);
}

// printar matrix
void print_matrix(double* mat, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%6.2f ", mat[w * i + j]);
        }
        printf("\n");
    }
}

image_double open_image(char* nome) {

    image_double struct_img;
    unsigned char *img = stbi_load(nome, &struct_img.w, &struct_img.h, &struct_img.c, 3);
    unsigned long int range = struct_img.h * struct_img.w;
    struct_img.R = (double*) malloc(range * sizeof(double));
    struct_img.G = (double*) malloc(range * sizeof(double));
    struct_img.B = (double*) malloc(range * sizeof(double));

    for (unsigned long int i = 0; i < struct_img.h; i++) {
        for (unsigned long int j = 0; j < struct_img.w; j++){
        int index = (i * struct_img.w + j) * 3;
            struct_img.R[struct_img.w * i + j] = (double) img[index];
            struct_img.G[struct_img.w * i + j] = (double) img[index + 1];
            struct_img.B[struct_img.w * i + j] = (double) img[index + 2];
        }
    }

    return struct_img;
}

image_char convert_from_double(image_double img) {
    image_char new_img;
    new_img.h = img.h;
    new_img.w = img.w;
    new_img.c = img.c;

    unsigned long int range_char = new_img.h * new_img.w;

    new_img.R = (unsigned char*) malloc(range_char * sizeof(unsigned char));
    new_img.G = (unsigned char*) malloc(range_char * sizeof(unsigned char));
    new_img.B = (unsigned char*) malloc(range_char * sizeof(unsigned char));

    for (int i = 0; i < new_img.h; i++){
        for (int j = 0; j < new_img.w; j++){
            new_img.R[new_img.w * i + j] = (unsigned char) round(img.R[img.w * i + j]);
            new_img.G[new_img.w * i + j] = (unsigned char) round(img.G[img.w * i + j]);
            new_img.B[new_img.w * i + j] = (unsigned char) round(img.B[img.w * i + j]);
        }
    }

    return new_img;
}


void free_image_double(image_double img) {
    free(img.R);
    free(img.G);
    free(img.B);
}

void free_image_char(image_char img){
    free(img.R);
    free(img.G);
    free(img.B);
}

void free_kernel(kernel Kernel){
    free(Kernel.kernel_values);
}