#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./headers/stb_image.h"
#include "./headers/stb_image_write.h"
#include <time.h>
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
//image_double apply_convolution_rgb(image_double img, kernel kernel_R, kernel kernel_G, kernel kernel_B); // Aplicar Convolução em RGB
image_double apply_convolution_rgb(image_double img, image_double out, double *Rp, double *Gp, double *Bp, kernel kernel_R, kernel kernel_G, kernel kernel_B);
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int, unsigned int, unsigned int, unsigned int, double sigma); // Aplicar Desfoque Gaussiano em RGB
image_double copy_image_rgb(image_double img); // Função para copiar imagem
void free_matrix(double* mat, int h); // Liberar Matriz
void print_matrix(double* mat, int h, int w); // Printar Matriz
image_double open_image(char* nome); // Abrir Imagem
image_char convert_from_double(image_double); // Converter de double para unsigned char
void free_image_double(image_double img); // Liberar Imagem
void free_image_char(image_char img); // Liberar Imagem
void free_kernel(kernel Kernel); // Liberar Kernel
void save_image(char* nome, image_double blurred); //Salvar imagem



int main(void) {

    double start, stop;
    char* input_path = "./images/image_2048.png"; // Arquivo de entrada
    char* output_path = "./images/outputserial.png"; // Arquivo de saída

    start = omp_get_wtime();
    // 1. Abrir a imagem de entrada.
    image_double img = open_image(input_path);
    stop = omp_get_wtime();
    printf("Tempo de abertura da imagem:: %f\n", stop-start);

    start = omp_get_wtime();
    // 2. Aplicar blur.
    unsigned int kernel_size_R = 5;
    unsigned int kernel_size_G = 5;
    unsigned int kernel_size_B = 5;
    unsigned int iterations = 100;
    double sigma = 1.0;

    image_double blurred = iterative_gaussian_blur_rgb(img, kernel_size_R, kernel_size_G, kernel_size_B, iterations, sigma);
    stop = omp_get_wtime();
    printf("Tempo de processamento do filtro Gaussiano: %f\n", stop-start);

    start = omp_get_wtime();
    save_image(output_path, blurred);
    stop = omp_get_wtime();
    printf("Tempo de salvamento da imagem:: %f\n", stop-start);

    // 6. liberar memória

    free_image_double(img);
    free_image_double(blurred);

    return 0;
}


kernel create_gaussian_kernel(int size, double sigma) {
    kernel new;
    new.side = size;
    int half = size / 2;
    double twoSigmaSqr = 2.0 * sigma * sigma;

    // alocar matriz 2D
    new.kernel_values = (double*) malloc(size * size * sizeof(double));

    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        int y = i - half;

        for (int j = 0; j < size; j++) {
            int x = j - half;

            double value = exp(-(x*x + y*y) / twoSigmaSqr);

            new.kernel_values[size * i + j] = value;
            sum += value;
        }
    }

    // normalizar (soma = 1)
    double invSum = 1.0 / sum; // evita várias divisões

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            new.kernel_values[size * i + j] *= invSum;
        }
    }

    return new;
}

image_double apply_convolution_rgb(image_double img, image_double out, double *Rp, double *Gp, double *Bp, kernel kernel_R, kernel kernel_G, kernel kernel_B){
    
    int pad_h_R = kernel_R.side / 2;
    int pad_w_R = kernel_R.side / 2;
    int pad_h_G = kernel_G.side / 2;
    int pad_w_G = kernel_G.side / 2;
    int pad_h_B = kernel_B.side / 2;
    int pad_w_B = kernel_B.side / 2;

    int padded_h_R = img.h + 2 * pad_h_R;
    int padded_w_R = img.w + 2 * pad_w_R;
    int padded_h_G = img.h + 2 * pad_h_G;
    int padded_w_G = img.w + 2 * pad_w_G;
    int padded_h_B = img.h + 2 * pad_h_B;
    int padded_w_B = img.w + 2 * pad_w_B;

    // preencher com replicação de borda
    for (int i = 0; i < padded_h_R; i++) {
        for (int j = 0; j < padded_w_R; j++) {

            int orig_i = i - pad_h_R;
            int orig_j = j - pad_w_R;

            // clamp
            if (orig_i < 0) orig_i = 0;
            if (orig_i >= img.h) orig_i = img.h - 1;
            if (orig_j < 0) orig_j = 0;
            if (orig_j >= img.w) orig_j = img.w - 1;

            Rp[padded_w_R * i + j] = img.R[img.w * orig_i + orig_j];
        }
    }
    for (int i = 0; i < padded_h_G; i++) {
        for (int j = 0; j < padded_w_G; j++) {

            int orig_i = i - pad_h_G;
            int orig_j = j - pad_w_G;

            // clamp
            if (orig_i < 0) orig_i = 0;
            if (orig_i >= img.h) orig_i = img.h - 1;
            if (orig_j < 0) orig_j = 0;
            if (orig_j >= img.w) orig_j = img.w - 1;

            Gp[padded_w_G * i + j] = img.G[img.w * orig_i + orig_j];
        }
    }
    for (int i = 0; i < padded_h_B; i++) {
        for (int j = 0; j < padded_w_B; j++) {

            int orig_i = i - pad_h_B;
            int orig_j = j - pad_w_B;

            // clamp
            if (orig_i < 0) orig_i = 0;
            if (orig_i >= img.h) orig_i = img.h - 1;
            if (orig_j < 0) orig_j = 0;
            if (orig_j >= img.w) orig_j = img.w - 1;

            Bp[padded_w_B * i + j] = img.B[img.w * orig_i + orig_j];
        }
    }

    // convolução (mesma lógica, só triplicada)
    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int ki = 0; ki < kernel_R.side; ki++) {
                for (int kj = 0; kj < kernel_R.side; kj++) {
                    double k_R = kernel_R.kernel_values[kernel_R.side * ki + kj];
                    sumR += Rp[padded_w_R * (i + ki) + (j + kj)] * k_R;
                }
            }
            for (int ki = 0; ki < kernel_G.side; ki++) {
                for (int kj = 0; kj < kernel_G.side; kj++) {
                    double k_G = kernel_G.kernel_values[kernel_G.side * ki + kj];
                    sumG += Gp[padded_w_G * (i + ki) + (j + kj)] * k_G;
                }
            }
            for (int ki = 0; ki < kernel_B.side; ki++) {
                for (int kj = 0; kj < kernel_B.side; kj++) {
                    double k_B = kernel_B.kernel_values[kernel_B.side * ki + kj];
                    sumB += Bp[padded_w_B * (i + ki) + (j + kj)] * k_B;
                }
            }

            out.R[out.w * i + j] = sumR;
            out.G[out.w * i + j] = sumG;
            out.B[out.w * i + j] = sumB;
        }
    }

    return out;
}

// iterativo
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int kernel_size_R, unsigned int kernel_size_G, unsigned int kernel_size_B, unsigned int iterations, double sigma) {

    // cria kernel uma vez
    kernel kernel_R = create_gaussian_kernel(kernel_size_R, sigma);
    kernel kernel_G = create_gaussian_kernel(kernel_size_G, sigma);
    kernel kernel_B = create_gaussian_kernel(kernel_size_B, sigma);

    // copia imagem inicial
    image_double current = copy_image_rgb(img);
    // image_double current = img;

    int pad_h_R = kernel_R.side / 2;
    int pad_w_R = kernel_R.side / 2;
    int pad_h_G = kernel_G.side / 2;
    int pad_w_G = kernel_G.side / 2;
    int pad_h_B = kernel_B.side / 2;
    int pad_w_B = kernel_B.side / 2;

    int padded_h_R = img.h + 2 * pad_h_R;
    int padded_w_R = img.w + 2 * pad_w_R;
    int padded_h_G = img.h + 2 * pad_h_G;
    int padded_w_G = img.w + 2 * pad_w_G;
    int padded_h_B = img.h + 2 * pad_h_B;
    int padded_w_B = img.w + 2 * pad_w_B;

    unsigned long int range_pad_R = padded_h_R * padded_w_R;
    unsigned long int range_pad_G = padded_h_G * padded_w_G;
    unsigned long int range_pad_B = padded_h_B * padded_w_B;

    // criar imagens com padding
    double* Rp = (double*) malloc(range_pad_R * sizeof(double));
    double* Gp = (double*) malloc(range_pad_G * sizeof(double));
    double* Bp = (double*) malloc(range_pad_B * sizeof(double));

    // saída
    image_double out;
    out.h = img.h;
    out.w = img.w;
    unsigned long int range_out = out.h * out.w;

    out.R = (double*) malloc(range_out * sizeof(double));
    out.G = (double*) malloc(range_out * sizeof(double));
    out.B = (double*) malloc(range_out * sizeof(double));

    for (unsigned int it = 0; it < iterations; it++) {

        out = apply_convolution_rgb(current, out ,Rp,Gp,Bp, kernel_R, kernel_G, kernel_B);

        // Troca os ponteiros
        image_double temp = current;
        current = out;
        out = temp;
    }

    //liberar padding
    free(Rp);
    free(Gp);
    free(Bp);

    // liberar kernel
    free_kernel(kernel_R);
    free_kernel(kernel_G);
    free_kernel(kernel_B);

    return current;
}

void save_image(char* output_path, image_double blurred){
    // 3. Converter de double para unsigned char pra saída.
    image_char out = convert_from_double(blurred);
    
    // 4. Montar buffer linear (RGB)
    unsigned char* buffer = malloc(out.w * out.h * 3);

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
    struct_img.R = malloc(range * sizeof(double));
    struct_img.G = malloc(range * sizeof(double));
    struct_img.B = malloc(range * sizeof(double));

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

    new_img.R = malloc(range_char * sizeof(unsigned char));
    new_img.G = malloc(range_char * sizeof(unsigned char));
    new_img.B = malloc(range_char * sizeof(unsigned char));

    for (int i = 0; i < new_img.h; i++){
        for (int j = 0; j < new_img.w; j++){
            new_img.R[new_img.w * i + j] = (unsigned char) round(img.R[img.w * i + j]);
            new_img.G[new_img.w * i + j] = (unsigned char) round(img.G[img.w * i + j]);
            new_img.B[new_img.w * i + j] = (unsigned char) round(img.B[img.w * i + j]);
        }
    }

    return new_img;
}

image_double copy_image_rgb(image_double img) {

    image_double copy;

    copy.h = img.h;
    copy.w = img.w;
    copy.c = img.c;
    unsigned long int range_copy = copy.h * copy.w;   
    copy.R = (double*) malloc(range_copy * sizeof(double));
    copy.G = (double*) malloc(range_copy * sizeof(double));
    copy.B = (double*) malloc(range_copy * sizeof(double));

    for (int i = 0; i < copy.h; i++) {
        for (int j = 0; j < copy.w; j++) {
            copy.R[copy.w * i + j] = img.R[img.w * i + j];
            copy.G[copy.w * i + j] = img.G[img.w * i + j];
            copy.B[copy.w * i + j] = img.B[img.w * i + j];
        }
    }

    return copy;
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