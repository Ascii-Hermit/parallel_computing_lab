#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolution1DKernel(const float* input, int width, const float* mask, int maskSize, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int maskRadius = maskSize / 2;

    if (i < width) {
        float sum = 0.0f;
        for (int j = 0; j < maskSize; ++j) {
            int inputIndex = i + j - maskRadius;
            if (inputIndex >= 0 && inputIndex < width) {
                sum += input[inputIndex] * mask[j];
            }
        }
        output[i] = sum;
    }
}

int main() {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    int width = sizeof(input) / sizeof(input[0]);

    float mask[] = {1.0f, 2.0f, 1.0f};
    int maskSize = sizeof(mask) / sizeof(mask[0]);

    float* output = (float*)malloc(width * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    float* d_input;
    float* d_mask;
    float* d_output;

    cudaMalloc(&d_input, width * sizeof(float));
    cudaMalloc(&d_mask, maskSize * sizeof(float));
    cudaMalloc(&d_output, width * sizeof(float));

    cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;
    convolution1DKernel<<<gridSize, blockSize>>>(d_input, width, d_mask, maskSize, d_output);

    cudaMemcpy(output, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int i = 0; i < width; ++i) {
        printf("%.2f ", input[i]);
    }
    printf("\n");

    printf("Mask: ");
    for (int i = 0; i < maskSize; ++i) {
        printf("%.2f ", mask[i]);
    }
    printf("\n");

    printf("Output: ");
    for (int i = 0; i < width; ++i) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    free(output);

    return 0;
}