#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int buffer_size = (size - 1) * sizeof(int) + MPI_BSEND_OVERHEAD; 

        void *buffer = malloc(buffer_size);
        MPI_Buffer_attach(buffer, buffer_size);

        for (int i = 1; i < size; i++) {
            int x = rand();
            MPI_Send(&rand, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        MPI_Buffer_detach(&buffer, &buffer_size);
        free(buffer);
    } else {
        int x;
        MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank & 1) 
            printf("Rank %d: %d\n", rank, x * x);
        else 
            printf("Rank %d: %d\n", rank, x * x * x);
    }

    MPI_Finalize();
    return 0;
}