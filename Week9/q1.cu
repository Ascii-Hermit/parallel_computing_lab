#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void spmv_csr_kernel(int *d_row_ptr, int *d_col_idx, float *d_values, float *d_x, float *d_y, int num_rows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = d_row_ptr[row];
        int row_end = d_row_ptr[row + 1];

        for (int i = row_start; i < row_end; ++i) {
            sum += d_values[i] * d_x[d_col_idx[i]];
        }
        d_y[row] = sum;
    }
}

void spmv_csr(int *h_row_ptr, int *h_col_idx, float *h_values, float *h_x, float *h_y, int num_rows, int num_nonzeros){
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_values, num_nonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, num_rows * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = TILE_SIZE;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    spmv_csr_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);

    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(){
    int num_rows = 4;
    int num_nonzeros = 6;
    int h_row_ptr[] = {0, 2, 4, 5, 6};  
    int h_col_idx[] = {0, 1, 0, 2, 3, 2};  
    float h_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  
    float h_x[] = {1.0, 2.0, 3.0, 4.0};  
    float h_y[4]; 

    spmv_csr(h_row_ptr, h_col_idx, h_values, h_x, h_y, num_rows, num_nonzeros);

    printf("Resulting vector y:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }

    return 0;
}
