#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void reduceWord(char* word, char* reducedWord, int wordLength) {
    int idx = threadIdx.x; 

    if (idx < wordLength) {
        for (int i = 0; i < wordLength - idx; i++) {
            reducedWord[idx * wordLength + i] = word[i];
        }
    }
}

int main() {
    char word[] = "AMBUJ SHUKLA";
    int wordLength = strlen(word);
    
    char *d_word, *d_reducedWord;
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_reducedWord, wordLength * wordLength * sizeof(char));

    cudaMemcpy(d_word, word, wordLength * sizeof(char), cudaMemcpyHostToDevice);

    reduceWord<<<1, wordLength>>>(d_word, d_reducedWord, wordLength);

    char reducedWord[wordLength * wordLength];
    
    cudaMemcpy(reducedWord, d_reducedWord, wordLength * wordLength * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Reduced word steps:\n");
    for (int i = 0; i < wordLength; i++) {
        for (int j = 0; j < wordLength - i; j++) {
            printf("%c", reducedWord[i * wordLength + j]);
        }
        if(i != wordLength-1)
            printf("->");
    }
    printf("\n");

    cudaFree(d_word);
    cudaFree(d_reducedWord);

    return 0;
}
