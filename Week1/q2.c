#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    if (rank%2==0) {
        printf("Rank: %d , Hello\n", rank);
    }
    else{
        printf("Rank: %d , World\n", rank);
    }
    MPI_Finalize();
    return 0;
}
