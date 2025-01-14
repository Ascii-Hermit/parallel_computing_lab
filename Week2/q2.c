#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int data = rand();
            MPI_Bsend(&data,1,MPI_INT,i, 0, MPI_COMM_WORLD);
        }
    } else {
        int number;

        MPI_Recv(&number,1,MPI_INT,0, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        printf("Process %d: %d\n", rank, number);
    }

    MPI_Finalize();
    return 0;
}