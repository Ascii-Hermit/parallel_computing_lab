#include <stdio.h>
#include <cuda_runtime.h>

#define N 16

__global__ void oddPhase(int *arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx % 2 == 1 && idx + 1 < N) {
        if (arr[idx] > arr[idx + 1]) {
          
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

__global__ void evenPhase(int *arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx % 2 == 0 && idx + 1 < N) {
        if (arr[idx] > arr[idx + 1]) {
           
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

void transposeSort(int *arr) {
    int *d_arr;

    cudaMalloc(&d_arr, N * sizeof(int));

    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;  
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < N; i++) {
        oddPhase<<<numBlocks, blockSize>>>(d_arr);
        cudaDeviceSynchronize();

        evenPhase<<<numBlocks, blockSize>>>(d_arr);
        cudaDeviceSynchronize();  
    }

    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    int arr[] = {64, 25, 12, 22, 11, 90, 42, 35, 27, 50, 2, 85, 71, 55, 91, 18};

    printf("Original Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    transposeSort(arr);

    printf("Sorted Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
