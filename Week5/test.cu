#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
    //printf("hello\n");
}

int main() {
    int N = 1000;
    int size = N * sizeof(int);

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    // Allocate memory on the host
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c = (int *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block and enough blocks to cover N elements
    add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // Copy the result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the first few results
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
