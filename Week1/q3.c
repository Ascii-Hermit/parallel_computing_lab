#include <stdio.h>
#include <mpi.h>

double perform_operation(double num1, double num2, char operation) {
    switch (operation) {
        case '+': return num1 + num2;
        case '-': return num1 - num2;
        case '*': return num1 * num2;
        case '/': 
            if (num2 != 0)
                return num1 / num2;
            else {
                printf("Error: Division by zero!\n");
                return 0.0;
            }
        default:
            printf("Invalid operation!\n");
            return 0.0;
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double num1, num2, local_result = 0.0, result = 0.0;
    char operation;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    num1 = 10;
    num2 = 12;
    operation = '+';
    MPI_Bcast(&num1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&operation, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_result = perform_operation(num1, num2, operation);
    MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final result of %.2f %c %.2f = %.2f\n", num1, operation, num2, result/10);
    }

    MPI_Finalize();
    return 0;
}
