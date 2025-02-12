#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // this is the unique id for each process
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank  == 0){
        printf("Hello from root\n");
    }
    else{
        printf("Hello from process %d of %d\n", rank, size);
    }

    MPI_Finalize();
    return 0;
}
