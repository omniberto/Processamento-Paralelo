#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../headers/stb_image.h"

int main(int argc, char** argv) {
    // Definindo os caminhos (voce pode alterar para os nomes reais dos seus arquivos)
    char* path_serial = "../images/outputserial.png";
    char* path_mpi = "../images/outputmpi.png";

    // Permite passar os arquivos via linha de comando (opcional)
    if (argc == 3) {
        path_serial = argv[1];
        path_mpi = argv[2];
    }

    printf("==================================================\n");
    printf("VERIFICADOR DE CORRETUDE - DESFOQUE GAUSSIANO\n");
    printf("==================================================\n");
    printf("Imagem Serial : %s\n", path_serial);
    printf("Imagem MPI    : %s\n\n", path_mpi);

    int w1, h1, c1;
    int w2, h2, c2;

    // Carrega as duas imagens
    unsigned char *img_serial = stbi_load(path_serial, &w1, &h1, &c1, 3);
    if (img_serial == NULL) {
        printf("ERRO: Nao foi possivel abrir a imagem serial.\n");
        return 1;
    }

    unsigned char *img_mpi = stbi_load(path_mpi, &w2, &h2, &c2, 3);
    if (img_mpi == NULL) {
        printf("ERRO: Nao foi possivel abrir a imagem MPI.\n");
        free(img_serial);
        return 1;
    }

    // 1. Verificacao de Dimensoes
    if (w1 != w2 || h1 != h2 || c1 != c2) {
        printf("ERRO GRAVE: As imagens possuem dimensoes ou canais diferentes!\n");
        printf("Serial: %d x %d (%d canais)\n", w1, h1, c1);
        printf("MPI   : %d x %d (%d canais)\n", w2, h2, c2);
        
        free(img_serial);
        free(img_mpi);
        return 1;
    }

    printf("Dimensoes conferem: %d x %d (%d canais)\n", w1, h1, c1);

    // 2. Verificacao Pixel a Pixel
    unsigned long int total_pixels = w1 * h1 * 3;
    unsigned long int pixels_divergentes = 0;
    int diferenca_maxima = 0;
    double erro_acumulado = 0.0;

    for (unsigned long int i = 0; i < total_pixels; i++) {
        // Calcula a diferenca absoluta entre o pixel do Serial e do MPI
        int diff = abs((int)img_serial[i] - (int)img_mpi[i]);

        if (diff > 0) {
            pixels_divergentes++;
            erro_acumulado += diff;
            
            if (diff > diferenca_maxima) {
                diferenca_maxima = diff;
            }
        }
    }

    // 3. Relatorio Final
    printf("\nRELATORIO DE COMPARACAO:\n");
    printf("--------------------------------------------------\n");
    printf("Total de pixels (canais RGB): %lu\n", total_pixels);
    printf("Pixels divergentes          : %lu\n", pixels_divergentes);
    printf("Diferenca Absoluta Maxima   : %d\n", diferenca_maxima);

    if (pixels_divergentes == 0) {
        printf("\nVEREDITO: SUCESSO ABSOLUTO!\n");
        printf("As imagens sao BIT A BIT IDENTICAS. O seu MPI esta perfeito!\n");
    } else {
        double erro_medio = erro_acumulado / total_pixels;
        printf("Erro medio por pixel        : %f\n", erro_medio);

        if (diferenca_maxima == 1) {
            printf("\nVEREDITO: ACEITAVEL (ARREDONDAMENTO).\n");
            printf("As imagens possuem pequenas diferencas, mas a diferenca maxima é de apenas 1 na escala RGB.\n");
            printf("Isso é normal devido ao arredondamento (round) do ponto flutuante em paralelismo.\n");
        } else {
            printf("\nVEREDITO: FALHA.\n");
            printf("As imagens sao visivelmente diferentes. Existe algum erro de logica (provavelmente nas Ghost Cells) no codigo MPI.\n");
        }
    }
    printf("==================================================\n");

    // Limpeza de memoria
    free(img_serial);
    free(img_mpi);

    return 0;
}