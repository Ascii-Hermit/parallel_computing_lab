


#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 5  // Define the array size

int main(int argc, char* argv[]) {
    int rank, size;
    int local_array[ARRAY_SIZE], global_array[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the local array with values based on rank
    for (int i = 0; i < ARRAY_SIZE; i++) {
        local_array[i] = rank + i;  // Example values: rank + index
    }

    // Perform element-wise sum using MPI_Reduce
    MPI_Reduce(local_array, global_array, ARRAY_SIZE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("Reduced array (element-wise sum): ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}







// MPI_Reduce(
//     void* send_data,
//     void* recv_data,
//     int count,
//     MPI_Datatype datatype,
//     MPI_Op op,
//     int root,
//     MPI_Comm communicator)