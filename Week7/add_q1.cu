#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void reverse_string_kernel(char *d_str, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n / 2) {
        char temp = d_str[idx];
        d_str[idx] = d_str[n - idx - 1];
        d_str[n - idx - 1] = temp;
    }
}

int main() {
    char str[1000];  
    
    printf("Enter a string: ");
    fgets(str, sizeof(str), stdin);
    
    str[strcspn(str, "\n")] = '\0';

    int n = strlen(str);  

    char *d_str;
    cudaMalloc((void**)&d_str, n * sizeof(char));

    cudaMemcpy(d_str, str, n * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n / 2 + blockSize - 1) / blockSize;

    reverse_string_kernel<<<numBlocks, blockSize>>>(d_str, n);
    cudaDeviceSynchronize();

    cudaMemcpy(str, d_str, n * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Reversed string: %s\n", str);

    cudaFree(d_str);

    return 0;
}
