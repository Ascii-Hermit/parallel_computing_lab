#include "mpi.h"
#include <stdio.h>
 
int main(int argc, char *argv[]) {
    int rank, size, num;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int A[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    if(MPI_Scatter(A, 1, MPI_INT, &num, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
        printf("Error in scattering");
        return 0;
    }
    int fact = 1;
    for (int i = num; i >= 1; i--) {
        fact *= i;
    }
    fprintf(stdout, "Received %d in process %d and its factorial is %d\n", num, rank, fact);
    fflush(stdout);
    int result = 0;
    if(MPI_Reduce(&fact, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
        printf("Error in reducing");
        return 0;
    }
    if (rank == 0) {
        printf("Sum of all factorials is: %d\n", result);
    }
    MPI_Finalize();
    return 0;
}