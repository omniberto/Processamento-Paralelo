#include <stdio.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>

#define SIZE 32

void free_matrix(double **mat, int h){
    for (int i = 0; i < h; i++){
        free(mat[i]);
    }
    free(mat);
}

int main(void) {
    srand(time(NULL));
    int w, h, c;

    unsigned char *img = stbi_load("image.png", &w, &h, &c, 3);

    double** R = malloc(h * sizeof(double*));
    double** G = malloc(h * sizeof(double*));
    double** B = malloc(h * sizeof(double*));

    for (int i = 0; i < h; i++) {
        R[i] = malloc(w * sizeof(double));
        G[i] = malloc(w * sizeof(double));
        B[i] = malloc(w * sizeof(double));

        for (int j = 0; j < w; j++) {
            int idx = (i * w +j) * 3;
            R[i][j] = img[idx];
            G[i][j] = img[idx + 1];
            B[i][j] = img[idx + 2];
        }
    }

    unsigned char* output = malloc(w * h * 3);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int idx = (i * w + j) * 3;
            output[idx] = ((int) (R[i][j]) + rand()) % 256;
            output[idx + 1] = ((int) (G[i][j]) + rand()) % 256;
            output[idx + 2] = ((int) (B[i][j]) + rand()) % 256;
        }
    }

    stbi_write_png("output.png", w, h, 3, output, w * 3);
    stbi_image_free(img);
    free_matrix(R, h);
    free_matrix(G, h);
    free_matrix(B, h);
    free(output);

    return 0;
}