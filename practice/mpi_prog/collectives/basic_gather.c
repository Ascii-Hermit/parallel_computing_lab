#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_value = rank * 2;  // Each process has a unique value (0, 2, 4, ...)
    int recv_values[size];  // Buffer at root to collect data

    MPI_Gather(&send_value, 1, MPI_INT, recv_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Root process gathered values: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", recv_values[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

// MPI_Gather(
//     const void *sendbuf,   // Pointer to send buffer (each process sends this)
//     int sendcount,         // Number of elements sent by each process
//     MPI_Datatype sendtype, // Data type of elements being sent
//     void *recvbuf,         // Pointer to receive buffer (only relevant for root)
//     int recvcount,         // Number of elements received from each process
//     MPI_Datatype recvtype, // Data type of received elements
//     int root,              // Rank of root process (which collects data)
//     MPI_Comm comm          // Communicator
// );


// int MPI_Gatherv(
//     const void *sendbuf,     // Local send buffer
//     int sendcount,           // Number of elements to send
//     MPI_Datatype sendtype,   // Type of each element in send buffer
//     void *recvbuf,           // Buffer in root process to receive data
//     const int *recvcounts,   // Array specifying how many elements each process sends
//     const int *displs,       // Array specifying displacements at which to place received data
//     MPI_Datatype recvtype,   // Type of each element in receive buffer
//     int root,                // Root process that gathers the data
//     MPI_Comm comm            // Communicator
// );
