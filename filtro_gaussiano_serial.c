#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers/stb_image.h"
#include "headers/stb_image_write.h"
#include <time.h>

typedef struct {
    double **R;
    double **G;
    double **B;
    int w;
    int h;
    int c;
} image_double;

typedef struct {
    unsigned char **R;
    unsigned char **G;
    unsigned char **B;
    int w;
    int h;
    int c;
} image_char;

double** create_gaussian_kernel(int size, double sigma);
double** apply_convolution(double** image, int img_h, int img_w, double** kernel, int k_h, int k_w);
double** iterative_gaussian_blur(double** image, int img_h, int img_w, int kernel_size, int iterations, double sigma);
double** copy_image(double** image, int h, int w);

void free_matrix(const void** mat, int h);
void print_matrix(double** mat, int h, int w);

image_double open_image(char* nome);
image_char convert_from_double(image_double);

int main(void){
    return 0;
}
double** create_gaussian_kernel(int size, double sigma) {
    int half = size / 2;
    double twoSigmaSqr = 2.0 * sigma * sigma;

    // alocar matriz 2D
    double** kernel = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        kernel[i] = (double*)malloc(size * sizeof(double));
    }

    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        int y = i - half;

        for (int j = 0; j < size; j++) {
            int x = j - half;

            double value = exp(-(x*x + y*y) / twoSigmaSqr);

            kernel[i][j] = value;
            sum += value;
        }
    }

    // normalizar (soma = 1)
    double invSum = 1.0 / sum; // evita várias divisões

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] *= invSum;
        }
    }

    return kernel;
}

double** apply_convolution(double** image, int img_h, int img_w, double** kernel, int k_h, int k_w) {

    int pad_h = k_h / 2;
    int pad_w = k_w / 2;

    int padded_h = img_h + 2 * pad_h;
    int padded_w = img_w + 2 * pad_w;

    // criar imagem com padding
    double** padded = (double**)malloc(padded_h * sizeof(double*));
    for (int i = 0; i < padded_h; i++) {
        padded[i] = (double*)malloc(padded_w * sizeof(double));
    }

    // preencher com replicação de borda
    for (int i = 0; i < padded_h; i++) {
        for (int j = 0; j < padded_w; j++) {

            int orig_i = i - pad_h;
            int orig_j = j - pad_w;

            // clamp (replicação de borda)
            if (orig_i < 0) orig_i = 0;
            if (orig_i >= img_h) orig_i = img_h - 1;
            if (orig_j < 0) orig_j = 0;
            if (orig_j >= img_w) orig_j = img_w - 1;

            padded[i][j] = image[orig_i][orig_j];
        }
    }

    // saída
    double** output = (double**)malloc(img_h * sizeof(double*));
    for (int i = 0; i < img_h; i++) {
        output[i] = (double*)malloc(img_w * sizeof(double));
    }

    // convolução
    for (int i = 0; i < img_h; i++) {
        for (int j = 0; j < img_w; j++) {

            double sum = 0.0;

            for (int ki = 0; ki < k_h; ki++) {
                for (int kj = 0; kj < k_w; kj++) {

                    sum += padded[i + ki][j + kj] * kernel[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }

    // liberar padded
    for (int i = 0; i < padded_h; i++) {
        free(padded[i]);
    }
    free(padded);

    return output;
}

// iterativo
double** iterative_gaussian_blur(double** image, int img_h, int img_w, int kernel_size, int iterations, double sigma) {

    // cria kernel uma vez
    double** kernel = create_gaussian_kernel(kernel_size, sigma);

    // copia imagem inicial
    double** current = copy_image(image, img_h, img_w);

    for (int it = 0; it < iterations; it++) {

        double** next = apply_convolution(current, img_h, img_w, kernel, kernel_size, kernel_size);

        // libera imagem anterior
        free_matrix(current, img_h);

        current = next;
    }

    // liberar kernel
    free_matrix(kernel, kernel_size);

    return current;
}

// libera matriz
void free_matrix(const void** mat, int h) {
    for (int i = 0; i < h; i++) {
        free(mat[i]);
    }
    free(mat);
}
// printar matrix
void print_matrix(double** mat, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%6.2f ", mat[i][j]);
        }
        printf("\n");
    }
}
image_double open_image(char* nome) {

    image_double struct_img;
    unsigned char *img = stbi_load(nome, &struct_img.w, &struct_img.h, &struct_img.c, 3);

    struct_img.R = malloc(struct_img.h * sizeof(double*));
    struct_img.G = malloc(struct_img.h * sizeof(double*));
    struct_img.B = malloc(struct_img.h * sizeof(double*));

    for (int i = 0; i < struct_img.h; i++) {
        struct_img.R[i] = malloc(struct_img.w * sizeof(double));
        struct_img.G[i] = malloc(struct_img.w * sizeof(double));
        struct_img.B[i] = malloc(struct_img.w * sizeof(double));

        for (int j = 0; j < struct_img.w; j++) {

            int index = (i * struct_img.w + j) * 3;
            struct_img.R[i][j] = (double) img[index];
            struct_img.G[i][j] = (double) img[index + 1];
            struct_img.B[i][j] = (double) img[index + 2];

        }
    }

    return struct_img;
}
image_char convert_from_double(image_double img) {
    image_char new_img;
    new_img.h = img.h;
    new_img.w = img.w;
    new_img.c = img.c;

    new_img.R = malloc(new_img.h * sizeof(unsigned char*));
    new_img.G = malloc(new_img.h * sizeof(unsigned char*));
    new_img.B = malloc(new_img.h * sizeof(unsigned char*));

    for (int i = 0; i < new_img.h; i++){
        new_img.R[i] = malloc(new_img.w * sizeof(unsigned char));
        new_img.G[i] = malloc(new_img.w * sizeof(unsigned char));
        new_img.B[i] = malloc(new_img.w * sizeof(unsigned char));

        for (int j = 0; j < new_img.w; j++){
            new_img.R[i][j] = (unsigned char) img.R[i][j];
            new_img.G[i][j] = (unsigned char) img.G[i][j];
            new_img.B[i][j] = (unsigned char) img.B[i][j];
        }
    }

    return new_img;
}