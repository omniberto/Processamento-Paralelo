#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double** create_gaussian_kernel(int size, double sigma);
double** apply_convolution(double** image, int img_h, int img_w, double** kernel, int k_h, int k_w);
double** iterative_gaussian_blur(double** image, int img_h, int img_w, int kernel_size, int iterations, double sigma);
double** copy_image(double** image, int h, int w);
void free_matrix(double** mat, int h);
void print_matrix(double** mat, int h, int w);

int main(void) {

    int h = 5, w = 5;

    // criar imagem exemplo
    double** image = (double**)malloc(h * sizeof(double*));
    for (int i = 0; i < h; i++) {
        image[i] = (double*)malloc(w * sizeof(double));
    }

    // preencher imagem (tipo sua matriz de exemplo)
    double data[5][5] = {
        {1, 2, 3, 2, 10},
        {2, 4, 6, 4, 2},
        {3, 6, 9, 6, 3},
        {2, 4, 6, 4, 2},
        {1, 2, 3, 2, 1}
    };

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            image[i][j] = data[i][j];

    printf("Imagem original:\n");
    print_matrix(image, h, w);

    // aplicar blur
    double** result = iterative_gaussian_blur(image, h, w, 3, 2, 1.0);

    printf("\nImagem apos blur:\n");
    print_matrix(result, h, w);

    // liberar memória
    free_matrix(image, h);
    free_matrix(result, h);

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

// copia matriz
double** copy_image(double** image, int h, int w) {
    double** copy = (double**)malloc(h * sizeof(double*));
    for (int i = 0; i < h; i++) {
        copy[i] = (double*)malloc(w * sizeof(double));
        for (int j = 0; j < w; j++) {
            copy[i][j] = image[i][j];
        }
    }
    return copy;
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