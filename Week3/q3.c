#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, i, len, nvc, tot;
    nvc = 0;
    tot = 0;
    char str1[25]; 
    char str2[25];  

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter Required String: \n");
        fgets(str1, 25, stdin); 

        len = strlen(str1);
        if (str1[len - 1] == '\n') {
            str1[len - 1] = '\0';
            len--;
        }
    }
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int chunk_size = len / size;
    MPI_Scatter(str1, chunk_size, MPI_CHAR, str2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (i = 0; i < chunk_size; i++) {
        char ch = str2[i];
        if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
            ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') {
            nvc++;
        }
    }
    int *all_vowel_counts = NULL;
    if (rank == 0) {
        all_vowel_counts = (int *)malloc(size * sizeof(int));
    }

    MPI_Gather(&nvc, 1, MPI_INT, all_vowel_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (i = 0; i < size; i++) {
            tot += all_vowel_counts[i];  
        }
        printf("Total number of vowels: %d\n", tot);
        free(all_vowel_counts);
    }

    MPI_Finalize();
    return 0;
}
