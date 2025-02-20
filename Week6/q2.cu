#include <stdio.h>
#include <cuda_runtime.h>

__global__ void findMinKernel(int *arr, int *result, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int res_ind = 0;
        for (int i = 0; i < N; i++) {
            if (arr[i] < val) {
                res_ind++;
            }
        }
        result[res_ind] = val;
    }
}

void printArray(int *arr, int N) {
    for (int i = 0; i < N; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N = 16; 
    int h_arr[] = {64, 25, 12, 22, 11, 90, 42, 35, 27, 50, 2, 85, 71, 55, 91, 18};
    
    int *d_arr, *d_result;
    int *h_result = new int[N]; 
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for(int i=0;i<N;i++){
        findMinKernel<<<gridSize, blockSize>>>(d_arr, d_result, h_arr[i] ,N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result Array:\n");
    printArray(h_result, N);

    cudaFree(d_arr);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}
