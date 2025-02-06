#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>  

__global__ void sine_kernel(float *d_input, float *d_output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        d_output[idx] = sinf(d_input[idx]); 
    }
}

int main() {
    int N = 1000; 
    int size = N * sizeof(float);

    float *h_input, *h_output;

    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i * 2 * M_PI / N);  
    }
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sine_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Sine of the first 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("sin(%f) = %f\n", h_input[i], h_output[i]);
    }
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
