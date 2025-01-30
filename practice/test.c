#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int n_elements = 10; 
    int chunk_size = n_elements / size; 
    int remainder = n_elements % size; 
    int start = rank * chunk_size; 
    int end = start + chunk_size + (rank == size-1 ? remainder:0);
    for (int i = start; i < end; i++) {
        local_sum += array[i];
    }
    printf("local sum is %d\n",local_sum);

    // Process 0 collects the sums from other processes
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only process 0 prints the final result
    if (rank == 0) {
        printf("Total sum: %d\n", total_sum);
    }

    MPI_Finalize(); 
    return 0;
}
