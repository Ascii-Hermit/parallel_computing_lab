#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_addition(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) { 
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000; 
    int size = N * sizeof(int); 
    int *h_A, *h_B, *h_C;
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;      
        h_B[i] = i * 3;  
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;  
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  

    vector_addition<<<N, threadsPerBlock>>>(d_A, d_B, d_C, N);
    vector_addition<<<blocksPerGrid, N>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
