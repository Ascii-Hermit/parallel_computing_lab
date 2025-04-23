#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sort(int *arr, int l_arr, int *ans) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i < l_arr) {
        int val = arr[i];
        int idx = 0;

        for (int j = 0; j < l_arr; ++j) {
            if (arr[j] < val || (arr[j] == val && j < i)) {
                idx += 1;
            }
        }

        ans[idx] = val;
    }
}

int main() {
    int *arr, *ans, n;

    printf("Enter size of array:\n");
    scanf("%d", &n);

    arr = (int *)malloc(sizeof(int) * n);
    ans = (int *)malloc(sizeof(int) * n);

    printf("Enter array elements:\n");
    for (int i = 0; i < n; ++i) {
        scanf("%d", &arr[i]);
        ans[i] = -1; 
    }

    int *darr, *dans;
    cudaMalloc((void **)&darr, n * sizeof(int));
    cudaMalloc((void **)&dans, n * sizeof(int));

    cudaMemcpy(darr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    sort<<<1, n>>>(darr, n, dans);
    cudaDeviceSynchronize();

    cudaMemcpy(ans, dans, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(darr);
    cudaFree(dans);

    printf("Sorted output:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", ans[i]);
    }
    printf("\n");

    free(arr);
    free(ans);

    return 0;
}
