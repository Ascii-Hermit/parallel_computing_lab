#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>


__global__ void add_vector(int* A, int* B, int* C, int N){
    int idx = threadIdx.x ;
    if(idx<N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    int N = 5;
    int* A, *B, *C;
    int* d_A, *d_B, *d_C;

    A = (int *)malloc(sizeof(int) * N);
    B = (int *)malloc(sizeof(int) * N);
    C = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++){
        A[i] = i + 1;
        B[i] = i * 3;
    }

    cudaMalloc((void **)&d_A, sizeof(int) * N);
    cudaMalloc((void **)&d_B, sizeof(int) * N);
    cudaMalloc((void **)&d_C, sizeof(int) * N);

    cudaMemcpy(d_A, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * N, cudaMemcpyHostToDevice);

    add_vector<<<1, N>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N;i++){
        printf("%d ", C[i]);
    }

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
// USING 2D THREADS //
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>


__global__ void add_vector(int* A, int* B, int* C, int N){
    int row = threadIdx.x + blockDim.x*blockIdx.x;
    int col = threadIdx.y + blockDim.y*blockIdx.y;

    if(row*N+col<N){
        C[row*N+col] = A[row*N+col] + B[row*N+col];
    }
}

int main(){
    int N = 5;
    int* A, *B, *C;
    int* d_A, *d_B, *d_C;

    A = (int *)malloc(sizeof(int) * N);
    B = (int *)malloc(sizeof(int) * N);
    C = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++){
        A[i] = i + 1;
        B[i] = i * 3;
    }

    cudaMalloc((void **)&d_A, sizeof(int) * N);
    cudaMalloc((void **)&d_B, sizeof(int) * N);
    cudaMalloc((void **)&d_C, sizeof(int) * N);

    cudaMemcpy(d_A, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * N, cudaMemcpyHostToDevice);

    dim3 blockDim(1, N);
    dim3 gridDim(1, 1);

    add_vector<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N;i++){
        printf("%d ", C[i]);
    }

    return 0;
}