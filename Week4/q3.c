#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat;
	int mat[4][4], ans[4][4];
	int temp1[4], temp2[4];
	if(rank == 0) {
		printf("Enter the 4x4 matrix:\n");
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				scanf("%d",&mat[i][j]);
			}
		}
	}
	MPI_Scatter(mat, 4, MPI_INT, temp1, 4, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scan(temp1, temp2, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Gather(temp2, 4, MPI_INT, ans[rank], 4, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank == 0) {
        printf("input\n");
        for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				printf("%d ", mat[i][j]);
			}
			printf("\n");
		}
        printf("output\n");
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				printf("%d ", ans[i][j]);
			}
			printf("\n");
		}
	}
	MPI_Finalize();
	exit(0);
}