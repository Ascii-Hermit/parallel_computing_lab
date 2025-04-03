#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct {
    char name[50];
    float price;
} Item;

const Item menu[] = {
    {"Laptop", 1200.0f},
    {"Smartphone", 800.0f},
    {"Headphones", 150.0f},
    {"Keyboard", 80.0f},
    {"Mouse", 50.0f},
    {"Charger", 30.0f}
};

__device__ const Item menuDevice[] = {
    {"Laptop", 1200.0f},
    {"Smartphone", 800.0f},
    {"Headphones", 150.0f},
    {"Keyboard", 80.0f},
    {"Mouse", 50.0f},
    {"Charger", 30.0f}
};

const int menuSize = sizeof(menu) / sizeof(menu[0]);

__global__ void calculateTotalPurchases(float* purchases, float* totalPurchases, int numFriends) {
    int friendId = blockIdx.x * blockDim.x + threadIdx.x;
    if (friendId < numFriends) {
        float total = 0.0f;
        int numItems = (int)purchases[friendId * (menuSize+1)];
        for (int i = 0; i < numItems; ++i) {
            for(int j = 0; j < menuSize; ++j){
                if(purchases[friendId * (menuSize+1) + j + 1] == 1.0f){
                    total += menuDevice[j].price;
                }
            }
        }
        totalPurchases[friendId] = total;
    }
}

int main() {
    int numFriends;
    printf("Enter the number of friends: ");
    scanf("%d", &numFriends);

    printf("\nShopping Mall Item Menu:\n");
    for (int i = 0; i < menuSize; ++i) {
        printf("%d. %s - $%.2f\n", i + 1, menu[i].name, menu[i].price);
    }

    float* purchases = (float*)malloc(numFriends * (menuSize + 1) * sizeof(float));
    if (purchases == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < numFriends; ++i) {
        printf("\nFriend %d purchases:\n", i + 1);
        int num_items = 0;
        printf("How many items did friend %d purchased?: ", i+1);
        scanf("%d", &num_items);
        purchases[i * (menuSize+1)] = (float)num_items;
        for (int j = 0; j < menuSize; ++j) {
            int buy;
            printf("Buy %s (1 for yes, 0 for no)? ", menu[j].name);
            scanf("%d", &buy);
            purchases[i * (menuSize + 1) + j + 1] = (float)buy;
        }
    }

    float* d_purchases;
    float* d_totalPurchases;
    cudaMalloc(&d_purchases, numFriends * (menuSize + 1) * sizeof(float));
    cudaMalloc(&d_totalPurchases, numFriends * sizeof(float));

    cudaMemcpy(d_purchases, purchases, numFriends * (menuSize + 1) * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numFriends + blockSize - 1) / blockSize;
    calculateTotalPurchases<<<gridSize, blockSize>>>(d_purchases, d_totalPurchases, numFriends);

    float* totalPurchases = (float*)malloc(numFriends * sizeof(float));
    if (totalPurchases == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        cudaFree(d_purchases);
        cudaFree(d_totalPurchases);
        free(purchases);
        return 1;
    }

    cudaMemcpy(totalPurchases, d_totalPurchases, numFriends * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nTotal Purchases:\n");
    for (int i = 0; i < numFriends; ++i) {
        printf("Friend %d: $%.2f\n", i + 1, totalPurchases[i]);
    }

    free(purchases);
    free(totalPurchases);
    cudaFree(d_purchases);
    cudaFree(d_totalPurchases);

    return 0;
}