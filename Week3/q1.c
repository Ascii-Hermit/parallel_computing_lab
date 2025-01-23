#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
long long factorial(int n) {
    long long prod = 1;
    for (int i = 1; i <= n; i++) {
        prod *= i;
    }
    return prod;
}

int main(int argc, char** argv) {
    int world_rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        int *values = (int *)malloc(world_size * sizeof(int));
    
        
        printf("Enter %d values:\n", world_size);
        for (int i = 0; i < world_size; i++) {
            printf("Value %d: ", i + 1);
            scanf("%d", &values[i]);
        }

        for (int i = 0; i < world_size; i++) {
            if (i < world_size) {
                MPI_Send(&values[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            } else {
                int dummy_value = 0;
                MPI_Send(&dummy_value, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        free(values);
    }

    int received_value;
    MPI_Recv(&received_value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    long long result = factorial(received_value);

    long long *all_results = NULL;
    if (world_rank == 0) {
        all_results = (long long *)malloc(world_size * sizeof(long long));

    }

    MPI_Gather(&result, 1, MPI_LONG_LONG, all_results, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        long long sum = 0;
        for (int i = 0; i < world_size; i++) {
            sum += all_results[i];
        }
        printf("Sum of all factorials: %lld\n", sum);
        free(all_results);
    }

    MPI_Finalize();
    return 0;
}
