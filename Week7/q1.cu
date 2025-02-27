#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_LENGTH 1000
#define MAX_WORDS 100

__global__ void count_word_kernel(char **words, int *counts, int num_words, const char *target_word, int target_word_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_words) {
        char *word = words[idx];
        int i = 0;
        
        int match = 1;
        for (i = 0; i < target_word_len && word[i] != '\0'; ++i) {
            if (word[i] != target_word[i]) {
                match = 0;
                break;
            }
        }

        if (match && word[i] == '\0' && target_word[i] == '\0') {
            atomicAdd(counts, 1);
        }
    }
}

int main() {
    char sentence[MAX_LENGTH];
    char word[MAX_LENGTH];
    char *words[MAX_WORDS];
    int counts[1] = {0};

    printf("Enter a sentence: ");
    fgets(sentence, sizeof(sentence), stdin);
    sentence[strcspn(sentence, "\n")] = '\0';

    printf("Enter the word to count: ");
    scanf("%s", word);

    char *token = strtok(sentence, " ,.-\n");
    int num_words = 0;
    while (token != NULL) {
        words[num_words] = token;
        num_words++;
        token = strtok(NULL, " ,.-\n");
    }

    char **d_words;
    int *d_counts;
    char *d_target_word;
    int target_word_len = strlen(word) + 1;

    cudaMalloc((void**)&d_words, num_words * sizeof(char*));
    cudaMalloc((void**)&d_counts, sizeof(int));
    cudaMalloc((void**)&d_target_word, target_word_len * sizeof(char));

    cudaMemcpy(d_target_word, word, target_word_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < num_words; i++) {
        char *d_word;
        cudaMalloc((void**)&d_word, strlen(words[i]) + 1);  
        cudaMemcpy(d_word, words[i], strlen(words[i]) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy(&d_words[i], &d_word, sizeof(char*), cudaMemcpyHostToDevice); 
    }

    int blockSize = 256;
    int numBlocks = (num_words + blockSize - 1) / blockSize;
    count_word_kernel<<<numBlocks, blockSize>>>(d_words, d_counts, num_words, d_target_word, target_word_len);

    cudaMemcpy(counts, d_counts, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", word, counts[0]);

    cudaFree(d_words);
    cudaFree(d_counts);
    cudaFree(d_target_word);

    return 0;
}
