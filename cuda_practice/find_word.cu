#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRINGS 5
#define MAX_LENGTH 100

// Constant memory for target word
__constant__ char target_word[MAX_LENGTH];

// Kernel to find the matching word
__global__ void findWord(char *str_arr, int *result, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < MAX_STRINGS) {
        char *current = str_arr + idx * stride;

        bool match = true;
        for (int i = 0; i < MAX_LENGTH; i++) {
            if (target_word[i] != current[i]) {
                match = false;
                break;
            }
            if (target_word[i] == '\0') break;
        }

        if (match) {
            *result = idx;
        }
    }
}

int main() {
    char str_arr[MAX_STRINGS][MAX_LENGTH];
    char search_term[MAX_LENGTH];
    printf("Enter %d strings:\n", MAX_STRINGS);

    for (int i = 0; i < MAX_STRINGS; i++) {
        fgets(str_arr[i], MAX_LENGTH, stdin);
        str_arr[i][strcspn(str_arr[i], "\n")] = '\0'; // Remove newline
    }

    printf("Enter word to search: ");
    fgets(search_term, MAX_LENGTH, stdin);
    search_term[strcspn(search_term, "\n")] = '\0';

    // Copy target word to constant memory
    cudaMemcpyToSymbol(target_word, search_term, MAX_LENGTH);

    // Allocate memory on device
    char *d_str_arr;
    int *d_result;
    int result = -1;

    cudaMalloc((void **)&d_str_arr, MAX_STRINGS * MAX_LENGTH);
    cudaMalloc((void **)&d_result, sizeof(int));
    cudaMemcpy(d_str_arr, str_arr, MAX_STRINGS * MAX_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    findWord<<<1, MAX_STRINGS>>>(d_str_arr, d_result, MAX_LENGTH);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    if (result != -1)
        printf("Match found at index: %d\n", result);
    else
        printf("No match found.\n");

    cudaFree(d_str_arr);
    cudaFree(d_result);

    return 0;
}
