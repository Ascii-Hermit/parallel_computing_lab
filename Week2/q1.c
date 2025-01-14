#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define max_length 100

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv); 

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		char string[] = "Hello world";

		// data, count, datatype, destination, tag, communicator
		MPI_Ssend(string, strlen(string) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		
		MPI_Recv(string, strlen(string) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		printf("Process 0: %s\n", string);
	} else if (rank == 1) {
		char string[max_length];

		MPI_Recv(string, max_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int i = 0; i < strlen(string); i++)
			string[i] = string[i] ^ 32;
		
		MPI_Ssend(string, strlen(string) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

	MPI_Finalize();
	return 0;
}