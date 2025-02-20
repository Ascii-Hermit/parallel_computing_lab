#include <stdio.h>
#include <cuda_runtime.h>
#define N 10  

__global__ void onesComplement(int *input, int *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < n) {
        output[idx] = ~input[idx];  
    }
}

void computeOnesComplement(int *input, int *output, int n) {
    int *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    onesComplement<<<numBlocks, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int output[N];

    printf("Original Array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");
    computeOnesComplement(input, output, N);

    printf("Array After Ones' Complement: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
