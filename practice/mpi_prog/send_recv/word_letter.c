#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

int main(){
    int rank,size;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    char str[25];
    char temp;

    if(rank == 0){
        printf("enter your string of len %d \n",size);
        fgets(str,25,stdin);
        temp = str[0];
        for(int i = 1;i<size;i++){
            MPI_Send(&str[i],1,MPI_CHAR,i,0,MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(&temp,1,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("%c",temp);
    char subs[25];
    for(int i = 0;i<rank+1;i++){
        subs[i] = temp;
    }
    //printf("%s\n",subs);
   int *recvbuf = NULL;
    int recvcounts[size];  // Array to store how many elements each process sends
    int displs[size];      // Array for displacements
int sendcount = rank + 1;
int sendbuf[sendcount];

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = i + 1;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];  // Cumulative sum
        }

        int total_size = displs[size - 1] + recvcounts[size - 1];
        recvbuf = (int *)malloc(total_size * sizeof(int));
    }

    // Gather variable-sized data
    MPI_Gatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Print collected data at root process
    if (rank == 0) {
        printf("Gathered array: ");
        for (int i = 0; i < displs[size - 1] + recvcounts[size - 1]; i++) {
            printf("%d ", recvbuf[i]);
        }
        printf("\n");
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}