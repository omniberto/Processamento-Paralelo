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
image_double apply_convolution_rgb(image_double img, double** kernel, int k_h, int k_w);
image_double iterative_gaussian_blur_rgb(image_double img, int kernel_size, int iterations, double sigma);
image_double copy_image_rgb(image_double img);
void free_matrix(double** mat, int h);
void print_matrix(double** mat, int h, int w);
image_double open_image(char* nome);
image_char convert_from_double(image_double);
void free_image(image_double img);


int main(void) {

    char* input_path = "images/image.png";
    char* output_path = "images/output.png";

    // 1. abrir imagem
    image_double img = open_image(input_path);

    printf("Imagem carregada: %dx%d\n", img.w, img.h);

    // 2. aplicar blur
    int kernel_size = 1;
    int iterations = 10;
    double sigma = 1.0;

    image_double blurred = iterative_gaussian_blur_rgb(img, kernel_size, iterations, sigma);

    // 3. converter usando sua função
    image_char out = convert_from_double(blurred);

    // 4. montar buffer linear (RGB)
    unsigned char* buffer = malloc(out.w * out.h * 3);

    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            int idx = (i * out.w + j) * 3;

            buffer[idx]     = out.R[i][j];
            buffer[idx + 1] = out.G[i][j];
            buffer[idx + 2] = out.B[i][j];
        }
    }

    // 5. salvar imagem
    stbi_write_png(output_path, out.w, out.h, 3, buffer, out.w * 3);

    printf("Imagem salva em: %s\n", output_path);

    // 6. liberar memória

    free_image(img);
    free_image(blurred);

    for (int i = 0; i < out.h; i++) {
        free(out.R[i]);
        free(out.G[i]);
        free(out.B[i]);
    }
    free(out.R);
    free(out.G);
    free(out.B);

    free(buffer);

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

image_double apply_convolution_rgb(image_double img, double** kernel, int k_h, int k_w) {

    int pad_h = k_h / 2;
    int pad_w = k_w / 2;

    int padded_h = img.h + 2 * pad_h;
    int padded_w = img.w + 2 * pad_w;

    // criar imagens com padding
    double** Rp = (double**)malloc(padded_h * sizeof(double*));
    double** Gp = (double**)malloc(padded_h * sizeof(double*));
    double** Bp = (double**)malloc(padded_h * sizeof(double*));

    for (int i = 0; i < padded_h; i++) {
        Rp[i] = (double*)malloc(padded_w * sizeof(double));
        Gp[i] = (double*)malloc(padded_w * sizeof(double));
        Bp[i] = (double*)malloc(padded_w * sizeof(double));
    }

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

            Rp[i][j] = img.R[orig_i][orig_j];
            Gp[i][j] = img.G[orig_i][orig_j];
            Bp[i][j] = img.B[orig_i][orig_j];
        }
    }

    // saída
    image_double out;
    out.h = img.h;
    out.w = img.w;

    out.R = (double**)malloc(out.h * sizeof(double*));
    out.G = (double**)malloc(out.h * sizeof(double*));
    out.B = (double**)malloc(out.h * sizeof(double*));

    for (int i = 0; i < out.h; i++) {
        out.R[i] = (double*)malloc(out.w * sizeof(double));
        out.G[i] = (double*)malloc(out.w * sizeof(double));
        out.B[i] = (double*)malloc(out.w * sizeof(double));
    }

    // convolução (mesma lógica, só triplicada)
    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int ki = 0; ki < k_h; ki++) {
                for (int kj = 0; kj < k_w; kj++) {

                    double k = kernel[ki][kj];

                    sumR += Rp[i + ki][j + kj] * k;
                    sumG += Gp[i + ki][j + kj] * k;
                    sumB += Bp[i + ki][j + kj] * k;
                }
            }

            out.R[i][j] = sumR;
            out.G[i][j] = sumG;
            out.B[i][j] = sumB;
        }
    }

    // liberar padding
    for (int i = 0; i < padded_h; i++) {
        free(Rp[i]);
        free(Gp[i]);
        free(Bp[i]);
    }
    free(Rp);
    free(Gp);
    free(Bp);

    return out;
}

// iterativo
image_double iterative_gaussian_blur_rgb(image_double img, int kernel_size, int iterations, double sigma) {

    // cria kernel uma vez
    double** kernel = create_gaussian_kernel(kernel_size, sigma);

    // copia imagem inicial
    image_double current = copy_image_rgb(img);

    for (int it = 0; it < iterations; it++) {

        image_double next = apply_convolution_rgb(current, kernel, kernel_size, kernel_size);

        // libera imagem anterior
        free_image(current);

        current = next;
    }

    // liberar kernel
    free_matrix(kernel, kernel_size);

    return current;
}

// libera matriz
void free_matrix(double** mat, int h) {
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

image_double copy_image_rgb(image_double img) {
    image_double copy;

    copy.h = img.h;
    copy.w = img.w;

    copy.R = (double**)malloc(copy.h * sizeof(double*));
    copy.G = (double**)malloc(copy.h * sizeof(double*));
    copy.B = (double**)malloc(copy.h * sizeof(double*));

    for (int i = 0; i < copy.h; i++) {
        copy.R[i] = (double*)malloc(copy.w * sizeof(double));
        copy.G[i] = (double*)malloc(copy.w * sizeof(double));
        copy.B[i] = (double*)malloc(copy.w * sizeof(double));

        for (int j = 0; j < copy.w; j++) {
            copy.R[i][j] = img.R[i][j];
            copy.G[i][j] = img.G[i][j];
            copy.B[i][j] = img.B[i][j];
        }
    }

    return copy;
}

void free_image(image_double img) {
    for (int i = 0; i < img.h; i++) {
        free(img.R[i]);
        free(img.G[i]);
        free(img.B[i]);
    }
    free(img.R);
    free(img.G);
    free(img.B);
}