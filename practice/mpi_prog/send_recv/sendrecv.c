#include <stdio.h>
#include <mpi.h>

int main(){
    MPI_Init(NULL,NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (rank == 0) {
        int number;
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) { // if this is a else statement, the proj will wait forever for the recv
        int recv;
        MPI_Recv(&recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n",
            recv);
    }
    MPI_Finalize();
}

// MPI_Send(
//     void* data,
//     int count,
//     MPI_Datatype datatype,
//     int destination,
//     int tag,
//     MPI_Comm communicator)

// MPI_Recv(
//     void* data,
//     int count,  ------> number of elements to send
//     MPI_Datatype datatype,
//     int source,
//     int tag,
//     MPI_Comm communicator,
//     MPI_Status* status)