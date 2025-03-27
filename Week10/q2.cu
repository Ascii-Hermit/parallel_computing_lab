#include <stdio.h>
#include <stdlib.h>

#define FILTER_SIZE 5  
#define BLOCK_SIZE 256 

__constant__ float d_filter[FILTER_SIZE];

__global__ void convolutionKernel(float *d_input, float *d_output, int inputSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < inputSize) {
        float result = 0.0f;
        
        for (int i = 0; i < FILTER_SIZE; i++) {
            int inputIndex = index + i - FILTER_SIZE / 2;
            if (inputIndex >= 0 && inputIndex < inputSize) {
                result += d_input[inputIndex] * d_filter[i];
            }
        }
        
        d_output[index] = result;
    }
}

void performConvolution(float *h_input, float *h_output, float *h_filter, int inputSize) {
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_output, inputSize * sizeof(float));

    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * sizeof(float));

    int gridSize = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(BLOCK_SIZE);

    convolutionKernel<<<dimGrid, dimBlock>>>(d_input, d_output, inputSize);

    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int inputSize = 10;  
    float *h_input, *h_output, *h_filter;

    h_input = (float*)malloc(inputSize * sizeof(float));
    h_output = (float*)malloc(inputSize * sizeof(float));
    h_filter = (float*)malloc(FILTER_SIZE * sizeof(float));

    for (int i = 0; i < inputSize; i++) {
        h_input[i] = i + 1.0f;  
    }
    
    for (int i = 0; i < FILTER_SIZE; i++) {
        h_filter[i] = 1.0f / FILTER_SIZE; 
    }

    performConvolution(h_input, h_output, h_filter, inputSize);

    for (int i = 0; i < 10; i++) {
        printf("Output[%d] = %f\n", i, h_output[i]);
    }

    free(h_input);
    free(h_output);
    free(h_filter);

    return 0;
}
