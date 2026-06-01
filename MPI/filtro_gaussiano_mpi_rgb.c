#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./headers/stb_image.h"
#include "./headers/stb_image_write.h"
#include <time.h>
#include <omp.h>
#include <mpi.h>

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
image_double apply_convolution_rgb(image_double img, image_double out, int rank, int nP, kernel kernel_R, kernel kernel_G, kernel kernel_B);
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int kernel_size_R, unsigned int kernel_size_G, unsigned int kernel_size_B, unsigned int iterations, double sigma, int rank, int nP); // Aplicar Desfoque Gaussiano em RGB
image_double copy_image_rgb(image_double img); // Função para copiar imagem
void free_matrix(double* mat, int h); // Liberar Matriz
void print_matrix(double* mat, int h, int w); // Printar Matriz
image_double open_image(char* nome); // Abrir Imagem
image_char convert_from_double(image_double); // Converter de double para unsigned char
void free_image_double(image_double img); // Liberar Imagem
void free_image_char(image_char img); // Liberar Imagem
void free_kernel(kernel Kernel); // Liberar Kernel
void save_image(char* nome, image_double blurred); //Salvar imagem



int main(int argc, char** argv) {

    double start, stop;
    char* input_path = "./images/image_2048.png"; // Arquivo de entrada
    char* output_path = "./images/outputmpi.png"; // Arquivo de saída
    int rank, nP, kernel_size = 5;
    image_double img;


    // Inicialização MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nP);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        start = omp_get_wtime();
        // Abrir a imagem de entrada.
        img = open_image(input_path);
        stop = omp_get_wtime();
        printf("Tempo de abertura da imagem:: %f\n", stop-start);
    }

    start = omp_get_wtime();

    // O Rank 0 precisa avisar aos outros processos as especificações da imagem
    int dimensions[3];
    if (rank == 0) {
        dimensions[0] = img.w;
        dimensions[1] = img.h;
        dimensions[2] = img.c;
    }
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        img.R = NULL;
        img.G = NULL;
        img.B = NULL;
        img.w = dimensions[0];
        img.h = dimensions[1];
        img.c = dimensions[2];
    }
    
    //Calcula quantas linhas cada processo precisa receber
    int local_h = img.h / nP;
    int remainder = img.h % nP;

    // Os primeiros processos recebem uma linha extra, caso a divisão não seja exata
    if (rank < remainder) {
        local_h++;
    }

    int pad = kernel_size / 2;

    // Determina o tamanho da margem com base na posição do processo
    int pad_top = (rank == 0) ? 0 : pad;
    int pad_bottom = (rank == nP - 1) ? 0 : pad;

    // Altura total alocada = pixels principais + bordas necessárias
    int alloc_h = local_h + pad_top + pad_bottom;

    image_double local_img;
    local_img.w = img.w;
    local_img.h = alloc_h; // A altura será apenas a parte da imagem atrelada ao processo
    local_img.c = img.c;

    // Alocamos a memória dos canais 
    local_img.R = malloc(local_img.h * local_img.w * sizeof(double));
    local_img.G = malloc(local_img.h * local_img.w * sizeof(double));
    local_img.B = malloc(local_img.h * local_img.w * sizeof(double));


    // Inicializa como NULL para todos os processos
    int *sendcounts = NULL;
    int *displs = NULL;

    // Apenas o Rank 0 aloca e calcula a distribuição
    if (rank == 0) {
        sendcounts = malloc(nP * sizeof(int));
        displs = malloc(nP * sizeof(int));
        int offset = 0;
        int rows_for_i;
        for (int i = 0; i < nP; i++) {
            rows_for_i = (img.h / nP) + (i < remainder ? 1 : 0);
            sendcounts[i] = rows_for_i * img.w;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    int recv_offset = pad_top * local_img.w;
 
    MPI_Scatterv(img.R, sendcounts, displs, MPI_DOUBLE, 
                 local_img.R + recv_offset, local_h * local_img.w, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(img.G, sendcounts, displs, MPI_DOUBLE, 
                 local_img.G + recv_offset, local_h * local_img.w, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(img.B, sendcounts, displs, MPI_DOUBLE, 
                 local_img.B + recv_offset, local_h * local_img.w, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Aplicar blur.
    unsigned int kernel_size_R = kernel_size;
    unsigned int kernel_size_G = kernel_size;
    unsigned int kernel_size_B = kernel_size;
    unsigned int iterations = 100;
    double sigma = 1.0;

    image_double blurred = iterative_gaussian_blur_rgb(local_img, kernel_size_R, kernel_size_G, kernel_size_B, iterations, sigma,rank,nP);
 
    int send_offset = pad_top * blurred.w;
    
    // A quantidade de dados enviados é apenas a parte sem padding (local_h)
    int send_count = local_h * blurred.w;

    // Reconstruindo a resposta
    MPI_Gatherv(blurred.R + send_offset, send_count, MPI_DOUBLE, img.R, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(blurred.G + send_offset, send_count, MPI_DOUBLE, img.G, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(blurred.B + send_offset, send_count, MPI_DOUBLE, img.B, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    stop = omp_get_wtime();

    if(rank == 0){
        printf("Tempo de processamento do filtro Gaussiano: %f\n", stop-start);

        start = omp_get_wtime();
        save_image(output_path, img);
        stop = omp_get_wtime();

        printf("Tempo de salvamento da imagem:: %f\n", stop-start);

        free_image_double(img);
        free(sendcounts);
        free(displs);
    }

    // liberar memória

    free_image_double(local_img);
    free_image_double(blurred);

    // Encerra o MPI
    MPI_Finalize();
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

image_double apply_convolution_rgb(image_double img, image_double out, int rank, int nP, kernel kernel_R, kernel kernel_G, kernel kernel_B){
    
    int pad = kernel_R.side / 2;

    int pad_top = (rank == 0) ? 0 : pad;
    int pad_bottom = (rank == nP - 1) ? 0 : pad;

    // O for percorre apenas a parte útil não os padding dos vizinhos
    int start_row = pad_top;
    int end_row = img.h - pad_bottom;

    // convolução (mesma lógica, só triplicada)
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < out.w; j++) {

            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int ki = 0; ki < kernel_R.side; ki++) {
                for (int kj = 0; kj < kernel_R.side; kj++) {
                    
                    // Calcular a coordenada real e fazer o clamp
                    int orig_i = i + ki - pad;
                    int orig_j = j + kj - pad;

                    if (orig_i < 0) orig_i = 0;
                    if (orig_i >= img.h) orig_i = img.h - 1;
                    if (orig_j < 0) orig_j = 0;
                    if (orig_j >= img.w) orig_j = img.w - 1;

                    double k_R = kernel_R.kernel_values[kernel_R.side * ki + kj];
                    sumR += img.R[img.w * orig_i + orig_j] * k_R;
                }
            }
            
            for (int ki = 0; ki < kernel_G.side; ki++) {
                for (int kj = 0; kj < kernel_G.side; kj++) {
                    
                    int orig_i = i + ki - pad;
                    int orig_j = j + kj - pad;

                    if (orig_i < 0) orig_i = 0;
                    if (orig_i >= img.h) orig_i = img.h - 1;
                    if (orig_j < 0) orig_j = 0;
                    if (orig_j >= img.w) orig_j = img.w - 1;

                    double k_G = kernel_G.kernel_values[kernel_G.side * ki + kj];
                    sumG += img.G[img.w * orig_i + orig_j] * k_G;
                }
            }
            
            for (int ki = 0; ki < kernel_B.side; ki++) {
                for (int kj = 0; kj < kernel_B.side; kj++) {
                    
                    int orig_i = i + ki - pad;
                    int orig_j = j + kj - pad;

                    if (orig_i < 0) orig_i = 0;
                    if (orig_i >= img.h) orig_i = img.h - 1;
                    if (orig_j < 0) orig_j = 0;
                    if (orig_j >= img.w) orig_j = img.w - 1;

                    double k_B = kernel_B.kernel_values[kernel_B.side * ki + kj];
                    sumB += img.B[img.w * orig_i + orig_j] * k_B;
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
image_double iterative_gaussian_blur_rgb(image_double img, unsigned int kernel_size_R, unsigned int kernel_size_G, unsigned int kernel_size_B, unsigned int iterations, double sigma, int rank, int nP) {

    // cria kernel uma vez
    kernel kernel_R = create_gaussian_kernel(kernel_size_R, sigma);
    kernel kernel_G = create_gaussian_kernel(kernel_size_G, sigma);
    kernel kernel_B = create_gaussian_kernel(kernel_size_B, sigma);

    // copia imagem inicial
    image_double current = copy_image_rgb(img);

    // Define o tamanho da margem (para kernel 5x5, pad = 2)
    int pad = kernel_R.side / 2; 

    // Calcula os pads específicos do rank
    int pad_top = (rank == 0) ? 0 : pad;
    int pad_bottom = (rank == nP - 1) ? 0 : pad;
    int top_neighbor = rank - 1;
    int bottom_neighbor = rank + 1;
    int offset_envio_baixo = (current.h - pad_bottom - pad) * current.w;
    int offset_recv_baixo = (current.h - pad_bottom) * current.w;

    // saída
    image_double out;
    image_double temp;
    out.h = img.h;
    out.w = img.w;
    unsigned long int range_out = out.h * out.w;

    out.R = (double*) malloc(range_out * sizeof(double));
    out.G = (double*) malloc(range_out * sizeof(double));
    out.B = (double*) malloc(range_out * sizeof(double));

    for (unsigned int it = 0; it < iterations; it++) {
        
        // Para evitar deadlock temos 2 fluxos
        // fluxo de cima pra baixo

        // Envia para o vizinho de BAIXO (exceto último rank)
        if (rank < nP - 1) {
            MPI_Send(current.R + offset_envio_baixo, pad * current.w, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD);
            MPI_Send(current.G + offset_envio_baixo, pad * current.w, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD);
            MPI_Send(current.B + offset_envio_baixo, pad * current.w, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD);
        }

        // Recebe do vizinho de CIMA (exceto 0)
        if (rank > 0) {
            MPI_Recv(current.R, pad_top * current.w, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(current.G, pad_top * current.w, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(current.B, pad_top * current.w, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // fluxo de baixo  pra cima
        // Envia para o vizinho de CIMA (se não for o Rank 0)
        // Usamos a Tag 1 aqui para diferenciar as mensagens anteriores
        if (rank > 0) {
            MPI_Send(current.R + (pad_top * current.w), pad * current.w, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD);
            MPI_Send(current.G + (pad_top * current.w), pad * current.w, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD);
            MPI_Send(current.B + (pad_top * current.w), pad * current.w, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD);
        }

        // D. Recebe do vizinho de BAIXO (exceto último rank))
        if (rank < nP - 1) {
            MPI_Recv(current.R + offset_recv_baixo, pad_bottom * current.w, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(current.G + offset_recv_baixo, pad_bottom * current.w, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(current.B + offset_recv_baixo, pad_bottom * current.w, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        out = apply_convolution_rgb(current, out ,rank,nP, kernel_R, kernel_G, kernel_B);

        // Troca os ponteiros
        temp = current;
        current = out;
        out = temp;
    }

    // liberar kernel
    free_kernel(kernel_R);
    free_kernel(kernel_G);
    free_kernel(kernel_B);

    free(out.R);
    free(out.G);
    free(out.B);

    return current;
}

void save_image(char* output_path, image_double blurred){
    // Converter de double para unsigned char pra saída.
    image_char out = convert_from_double(blurred);
    
    // ontar buffer linear (RGB)
    unsigned char* buffer = malloc(out.w * out.h * 3);

    for (int i = 0; i < out.h; i++) {
        for (int j = 0; j < out.w; j++) {

            int idx = (i * out.w + j) * 3;

            buffer[idx]     = out.R[out.w * i + j];
            buffer[idx + 1] = out.G[out.w * i + j];
            buffer[idx + 2] = out.B[out.w * i + j];
        }
    }

    // Salvar imagem
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