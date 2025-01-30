#include "mpi.h"
#include <stdio.h>
int main(int argc, char *argv[]) {
    int rank, size, m, ele;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) {
        printf("Enter M: ");
        scanf("%d", &m);   
        printf("Enter element to be searched");
        scanf("%d",&ele);     
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ele, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int A[size * m],C[m];
    if(rank == 0) {
        printf("Enter %d values\n", size * m);
        for(int i = 0; i < size * m; i++)
            scanf("%d", &A[i]);
    }
    MPI_Scatter(A, m, MPI_INT, C, m, MPI_INT, 0, MPI_COMM_WORLD);
    int freq = 0;
    for(int i = 0; i < m; i++) {
        freq+=(C[i] == ele ?1:0);
    }
    fprintf(stdout, "Found %d occurences in process %d \n", freq, rank);
    fflush(stdout);
    int result = 0;
    MPI_Reduce(&freq, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        printf("Total frequncy is: %d\n", result);
    }
    MPI_Finalize();
    return 0;
}