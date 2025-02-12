#include <stdio.h>
#include <string.h>
#include <mpi.h>

int isVowel(char ch){
    if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
        ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') {
        return 1;
    }
    return 0;
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, size, len = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char str1[25];
    char str2[25];

    if (rank == 0) {
        printf("Enter your string:\n");
        fgets(str1, 25, stdin);
        
        len = strlen(str1);
        if (str1[len - 1] == '\n') {
            str1[len - 1] = '\0';
            len--;
        }
    }
    
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {    
        for (int i = 1; i < size; i++) {
            MPI_Send(str1, len + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);    
        }
        strcpy(str2, str1);
    } else {
        MPI_Recv(str2, len + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    printf("Process %d got string: %s\n", rank, str2);
    strcpy(str2, "");
    
    int chunk = len / size;
    int remainder = len % size;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int start = i * chunk;
            int chunk_size = chunk;
            if (i == size - 1) {
                chunk_size += remainder;
            }
            MPI_Send(&str1[start], chunk_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
        strncpy(str2, str1, chunk);
        str2[chunk] = '\0';
    } else {
        int chunk_size = chunk;
        if (rank == size - 1) {
            chunk_size += remainder;
        }
        MPI_Recv(&str2, chunk_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        str2[chunk_size] = '\0';
    }
    printf("## Process %d got string segment: %s\n", rank, str2);

    for(int i = 0;i<strlen(str2);i++){
        if(isVowel(str2[i])){
            str2[i] = '*';
        }
    }
    char gathered_str[len + 1];
    MPI_Gather(str2, chunk , MPI_CHAR, gathered_str, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        gathered_str[len] = '\0';
        printf("Edited string after replacing vowels: %s\n", gathered_str);
    }
    
    MPI_Finalize();
    return 0;
}
