#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers/stb_image.h"
#include "headers/stb_image_write.h"
#include <time.h>

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

double* create_gaussian_kernel(int size, double sigma); // Criar Kernel
image_double apply_convolution_rgb(image_double img, double* kernel, int k_h, int k_w); // Aplicar Convolução em RGB
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int kernel_size, unsigned int iterations, double sigma); // Aplicar Desfoque Gaussiano em RGB
image_double copy_image_rgb(image_double img); // Função para copiar imagem
void free_matrix(double* mat, int h); // Liberar Matriz
void print_matrix(double* mat, int h, int w); // Printar Matriz
image_double open_image(char* nome); // Abrir Imagem
image_char convert_from_double(image_double); // Converter de double para unsigned char
void free_image_double(image_double img); // Liberar Imagem
void free_image_char(image_char img); // Liberar Imagem


int main(void) {

    char* input_path = "images/image.png"; // Arquivo de entrada
    char* output_path = "images/outputrgb.png"; // Arquivo de saída

    // 1. Abrir a imagem de entrada.
     image_double img = open_image(input_path);
    /*image_double img;
    double data[5][5] = {
        {255, 210, 30, 225, 111},
        {222, 254, 156, 42, 222},
        {13, 46, 59, 56, 38},
        {212, 154, 456, 49, 205},
        {105, 205, 35, 225, 115}
    };

    img.h = 5;
    img.w = 5;
    img.c = 3;
    img.R = malloc(img.h * sizeof(double*));
    img.G = malloc(img.h * sizeof(double*));
    img.B = malloc(img.h * sizeof(double*));

    for(int i = 0; i < img.h; i++) {
        img.R[i] = malloc(img.w * sizeof(double));
        img.G[i] = malloc(img.w * sizeof(double));
        img.B[i] = malloc(img.w * sizeof(double));
        for(int j = 0; j < img.w; j++){
            img.R[i][j] = data[i][j];
            img.G[i][j] = data[i][j];
            img.B[i][j] = data[i][j];
        }
    }
    */

    // 2. Aplicar blur.
    unsigned int kernel_size = 3;
    unsigned int iterations = 2;
    double sigma = 1.0;

    image_double blurred = iterative_gaussian_blur_rgb(img, kernel_size, iterations, sigma);
    printf("Verificao final:\n");
    for (int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%.2lf ", blurred.R[blurred.h *i + j]);
        }
        printf("\n");
    }
    printf("\n");

    // 3. Converter de double para unsigned char pra saída.
    image_char out = convert_from_double(blurred);
    
    // 4. Montar buffer linear (RGB)
    unsigned char* buffer = malloc(out.w * out.h * 3);

    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            int idx = (i * out.w + j) * 3;

            buffer[idx]     = out.R[out.h * i + j];
            buffer[idx + 1] = out.G[out.h * i + j];
            buffer[idx + 2] = out.B[out.h * i + j];
        }
    }
    printf("Verificao final:\n");
    for (int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            printf("%i ", out.R[out.h * i + j]);
        }
        printf("\n");
    }
    printf("\n");
    // 5. salvar imagem
    stbi_write_png(output_path, out.w, out.h, 3, buffer, out.w * 3);
    printf("Imagem salva em: %s\n", output_path);

    // 6. liberar memória

    free_image_double(img);
    free_image_double(blurred);
    free_image_char(out);
    free(buffer);

    return 0;
}

double* create_gaussian_kernel(int size, double sigma) {
    int half = size / 2;
    double twoSigmaSqr = 2.0 * sigma * sigma;

    // alocar matriz 2D
    double* kernel = (double*) malloc(size * size * sizeof(double));

    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        int y = i - half;

        for (int j = 0; j < size; j++) {
            int x = j - half;

            double value = exp(-(x*x + y*y) / twoSigmaSqr);

            kernel[size * i + j] = value;
            sum += value;
        }
    }

    // normalizar (soma = 1)
    double invSum = 1.0 / sum; // evita várias divisões

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[size * i + j] *= invSum;
        }
    }

    return kernel;
}

image_double apply_convolution_rgb(image_double img, double* kernel, int k_h, int k_w) {

    int pad_h = k_h / 2;
    int pad_w = k_w / 2;

    int padded_h = img.h + 2 * pad_h;
    int padded_w = img.w + 2 * pad_w;
    unsigned long int range_pad = padded_h * padded_w;

    // criar imagens com padding
    double* Rp = (double*) malloc(range_pad * sizeof(double));
    double* Gp = (double*) malloc(range_pad * sizeof(double));
    double* Bp = (double*) malloc(range_pad * sizeof(double));

    // preencher com replicação de borda
    for (int i = 0; i < padded_h; i++) {
        for (int j = 0; j < padded_w; j++) {

            int orig_i = i - pad_h;
            int orig_j = j - pad_w;

            // clamp
            if (orig_i < 0) orig_i = 0;
            if (orig_i >= img.h) orig_i = img.h - 1;
            if (orig_j < 0) orig_j = 0;
            if (orig_j >= img.w) orig_j = img.w - 1;

            Rp[padded_h * i + j] = img.R[img.h * orig_i + orig_j];
            Gp[padded_h * i + j] = img.G[img.h * orig_i + orig_j];
            Bp[padded_h * i + j] = img.B[img.h * orig_i + orig_j];
        }
    }

    // saída
    image_double out;
    out.h = img.h;
    out.w = img.w;
    unsigned long int range_out = out.h * out.w;

    out.R = (double*) malloc(range_out * sizeof(double));
    out.G = (double*) malloc(range_out * sizeof(double));
    out.B = (double*) malloc(range_out * sizeof(double));

    // convolução (mesma lógica, só triplicada)
    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int ki = 0; ki < k_h; ki++) {
                for (int kj = 0; kj < k_w; kj++) {

                    double k = kernel[k_h * ki + kj];

                    sumR += Rp[padded_h * (i + ki) + (j + kj)] * k;
                    sumG += Gp[padded_h * (i + ki) + (j + kj)] * k;
                    sumB += Bp[padded_h * (i + ki) + (j + kj)] * k;
                }
            }

            out.R[out.h * i + j] = sumR;
            out.G[out.h * i + j] = sumG;
            out.B[out.h * i + j] = sumB;
        }
    }

    // liberar padding
    free(Rp);
    free(Gp);
    free(Bp);

    return out;
}

// iterativo
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int kernel_size, unsigned int iterations, double sigma) {

    // cria kernel uma vez
    double* kernel = create_gaussian_kernel(kernel_size, sigma);

    // copia imagem inicial
    image_double current = copy_image_rgb(img);
    // image_double current = img;

    for (unsigned int it = 0; it < iterations; it++) {

        image_double next = apply_convolution_rgb(current, kernel, kernel_size, kernel_size);

        // libera imagem anterior
        free_image_double(current);

        current = next;
    }

    // liberar kernel
    free(kernel);

    return current;
}

// libera matriz
void free_matrix(double* mat, int h) {
    free(mat);
}

// printar matrix
void print_matrix(double* mat, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%6.2f ", mat[h * i + j]);
        }
        printf("\n");
    }
}

image_double open_image(char* nome) {

    image_double struct_img;
    unsigned char *img = stbi_load(nome, &struct_img.w, &struct_img.h, &struct_img.c, 3);
    unsigned long int range = struct_img.h * struct_img.w;
    struct_img.R = malloc(range * sizeof(double));
    struct_img.G = malloc(range * sizeof(double*));
    struct_img.B = malloc(range * sizeof(double*));

    for (unsigned long int i = 0; i < struct_img.h; i++) {
        for (unsigned long int j = 0; j < struct_img.w; j++){
        int index = (i * struct_img.w + j) * 3;
            struct_img.R[struct_img.h * i + j] = (double) img[index];
            struct_img.G[struct_img.h * i + j] = (double) img[index + 1];
            struct_img.B[struct_img.h * i + j] = (double) img[index + 2];
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
            new_img.R[new_img.h * i + j] = (unsigned char) round(img.R[img.h * i + j]);
            new_img.G[new_img.h * i + j] = (unsigned char) round(img.G[img.h * i + j]);
            new_img.B[new_img.h * i + j] = (unsigned char) round(img.B[img.h * i + j]);
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
            copy.R[copy.h * i + j] = img.R[img.h * i + j];
            copy.G[copy.h * i + j] = img.G[img.h * i + j];
            copy.B[copy.h * i + j] = img.B[img.h * i + j];
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