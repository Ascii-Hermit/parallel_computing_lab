#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int number;

    if (rank == 0) {
        printf("Enter a number to broadcast: ");
        scanf("%d", &number);
    }
    
    
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received number: %d\n", rank, number);

    MPI_Finalize();
    return 0;
}

// MPI_Bcast(
//     void* data,
//     int count,
//     MPI_Datatype datatype,
//     int root,
//     MPI_Comm communicator)