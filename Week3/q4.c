#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, len;
    char S1[100], S2[100], result[200];
    char *sub_S1, *sub_S2, *sub_result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the first string (S1): ");
        scanf("%s", S1);
        printf("Enter the second string (S2): ");
        scanf("%s", S2);

        len = strlen(S1);

        if (len % size != 0) {
            printf("The length of the strings must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    sub_S1 = (char *)malloc(len / size * sizeof(char));
    sub_S2 = (char *)malloc(len / size * sizeof(char));
    sub_result = (char *)malloc(len / size * 2 * sizeof(char)); 

    MPI_Scatter(S1, len / size, MPI_CHAR, sub_S1, len / size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, len / size, MPI_CHAR, sub_S2, len / size, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < len / size; i++) {
        sub_result[2 * i] = sub_S1[i]; 
        sub_result[2 * i + 1] = sub_S2[i]; 
    }

    MPI_Gather(sub_result, len, MPI_CHAR, result, len, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result[len] = '\0'; 
        printf("Resultant String: %s\n", result);
    }

    free(sub_S1);
    free(sub_S2);
    free(sub_result);

    MPI_Finalize();
    return 0;
}
