#include <stdio.h>
#include <mpi.h>

double pow_mpi(double base, int power) {
    if (power == 0) return 1.0;
    if (power == 1) return base;

    double result = 1.0;
    for (int i = 0; i < power; i++) {
        result *= base;
    }
    return result;
}

int main(int argc, char **argv) {
    int rank, size;
    double base = 2.0;  
    int power = 9;    
    double result = 1.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Bcast(&power, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rem = power % size; //9
    int local_power = power / size;  //0

    if (rank < rem) {
        local_power += 1;
    }

    double local_result = pow_mpi(base, local_power); 
    printf("Process %d (out of %d) computed part of result: %.2f\n", rank, size, local_result);
    MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final result of %.2f^%d = %.2f\n", base, power, result);
    }
    MPI_Finalize();
    return 0;
}
