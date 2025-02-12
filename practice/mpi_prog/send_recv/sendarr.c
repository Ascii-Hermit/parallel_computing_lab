#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 0;
    if (rank == 0) {
        printf("Enter size of array: \n");
        scanf("%d",&N);
        int *arr = (int *)malloc(N * sizeof(int));
        MPI_Send(&N, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        printf("Enter %d elements\n",N);
        for(int i = 0;i<N;i++){ 
            scanf("%d",&arr[i]);

        }
        MPI_Send(arr, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent entire array to Process 1\n");
    } 
    else if (rank == 1) {
        MPI_Recv(&N, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int* received_arr = (int *)malloc(N * sizeof(int)); 
        MPI_Recv(received_arr, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 1 received array: {");
        for (int i = 0; i < N; i++) {
            printf("%d ", received_arr[i]);
        }
        printf("}\n");
    }
    MPI_Finalize();
    return 0;
}
