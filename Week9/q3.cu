#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  
__global__ void modify_matrix_kernel(int *d_matrix, int num_rows, int num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= 1 && row < num_rows - 1 && col >= 1 && col < num_cols - 1) {
        d_matrix[row * num_cols + col] = ~d_matrix[row * num_cols + col];
    }
}

void int_to_binary(int num, char *binary_str)
{
    for (int i = 4; i >= 0; i--) {  
        binary_str[4 - i] = (num & (1 << i)) ? '1' : '0';
    }
    binary_str[5] = '\0';
}

void modify_matrix(int *h_matrix, int num_rows, int num_cols)
{
    int *d_matrix;
    
    cudaMalloc((void**)&d_matrix, num_rows * num_cols * sizeof(int));
    
    cudaMemcpy(d_matrix, h_matrix, num_rows * num_cols * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((num_rows + TILE_SIZE - 1) / TILE_SIZE, (num_cols + TILE_SIZE - 1) / TILE_SIZE);
    
    modify_matrix_kernel<<<gridSize, blockSize>>>(d_matrix, num_rows, num_cols);
    
    cudaMemcpy(h_matrix, d_matrix, num_rows * num_cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
}

int main()
{
    int num_rows = 5;
    int num_cols = 5;
    
    int h_matrix[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    printf("Original Matrix:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%d ", h_matrix[i * num_cols + j]);
        }
        printf("\n");
    }

    modify_matrix(h_matrix, num_rows, num_cols);

    printf("\nModified Matrix:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (i >= 1 && i < num_rows - 1 && j >= 1 && j < num_cols - 1) {
                char binary_str[6];  
                int_to_binary(h_matrix[i * num_cols + j], binary_str);
                printf("%s ", binary_str);  
            } else {
                printf("%d ", h_matrix[i * num_cols + j]);
            }
        }
        printf("\n");
    }

    return 0;
}
