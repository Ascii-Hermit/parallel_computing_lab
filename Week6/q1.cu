#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(int *n, int *m, int *output, int width, int m_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= m_size / 2 && idx < width - m_size / 2) {
        float conv_value = 0.0f;

        for (int i = 0; i < m_size; i++) {
            conv_value += n[idx + i - m_size / 2] * m[i];
        }

        output[idx] = conv_value;
    }
}

void convolution_1d(int *n, int *m, int *output, int width, int m_size) {
    int *d_n, *d_m, *d_output;
    cudaMalloc(&d_n, width * sizeof(int));
    cudaMalloc(&d_m, m_size * sizeof(int));
    cudaMalloc(&d_output, width * sizeof(int));

    cudaMemcpy(d_n, n, width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, m_size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;

    convolution_1d_kernel<<<numBlocks, blockSize>>>(d_n, d_m, d_output, width, m_size);
    cudaMemcpy(output, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_output);
}

int main() {
    int width = 10;
    int m_size = 3;

    int n[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int m[] = {1, 2, 3}; 
    int output[10] = {0}; 

    convolution_1d(n, m, output, width, m_size);

    printf("Convolution Output: ");
    for (int i = 0; i < width; i++) {
        printf("%d ",output[i]);
    }
    printf("\n");
    return 0;
}
