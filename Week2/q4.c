#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main() {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // pass a number and wait for it to come back
        int x;
        printf("Enter a number: \n");
        scanf("%d", &x);

        MPI_Send(&x, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 0 received %d\n", x);
    } else {
        // receive a number and pass it forward
        int x;
        MPI_Recv(&x, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d received %d\n", rank, x);
        
        x += 1;
        int next_rank = (rank + 1) % size;
        MPI_Send(&x, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}