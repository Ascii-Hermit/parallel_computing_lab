#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16 
__global__ void modify_matrix_kernel(float *d_matrix, int num_rows, int num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols)
    {
        int row_index = row + 1;  
        
        d_matrix[row * num_cols + col] = powf(d_matrix[row * num_cols + col], row_index);
    }
}

void modify_matrix(float *h_matrix, int num_rows, int num_cols)
{
    float *d_matrix;
    
    cudaMalloc((void**)&d_matrix, num_rows * num_cols * sizeof(float));
    
    cudaMemcpy(d_matrix, h_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((num_rows + TILE_SIZE - 1) / TILE_SIZE, (num_cols + TILE_SIZE - 1) / TILE_SIZE);
    
    modify_matrix_kernel<<<gridSize, blockSize>>>(d_matrix, num_rows, num_cols);
    
    cudaMemcpy(h_matrix, d_matrix, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
}

int main()
{
    int num_rows = 4;
    int num_cols = 4;
    
    float h_matrix[16] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

    printf("Original Matrix:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%.2f ", h_matrix[i * num_cols + j]);
        }
        printf("\n");
    }

    modify_matrix(h_matrix, num_rows, num_cols);

    printf("\nModified Matrix:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%.2f ", h_matrix[i * num_cols + j]);
        }
        printf("\n");
    }

    return 0;
}
