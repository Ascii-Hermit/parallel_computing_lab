#include <stdio.h>
#include <mpi.h>

long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

long long fibonacci(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main(int argc, char **argv) {
    int rank, size;
    long long result = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (i % 2 == 0) {
                printf("Rank %d -> Factorial: %lld\n", i, factorial(i));
            } else {
                printf("Rank %d -> Fibonacci: %lld\n", i, fibonacci(i));
            }
        }
    }
    MPI_Finalize();
    return 0;
}
