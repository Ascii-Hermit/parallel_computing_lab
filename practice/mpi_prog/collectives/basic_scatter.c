// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>

// int main(int argc, char* argv[]) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int elements_per_process = 2;  // Each process gets 2 elements
//     int *data = NULL;              // Pointer for dynamic allocation
//     int *recv_nums = (int*)malloc(elements_per_process * sizeof(int));  // Buffer for received data

//     // Only rank 0 initializes the full array dynamically
//     if (rank == 0) {
//         int total_elements = size * elements_per_process;
//         data = (int*)malloc(total_elements * sizeof(int));

//         printf("Original data: ");
//         for (int i = 0; i < total_elements; i++) {
//             data[i] = i + 1;  // Example: {1, 2, 3, 4, 5, 6, 7, 8}
//             printf("%d ", data[i]);
//         }
//         printf("\n");
//     }

//     // Scatter: Send chunks of 2 elements to each process
//     MPI_Scatter(data, elements_per_process, MPI_INT,
//                 recv_nums, elements_per_process, MPI_INT,
//                 0, MPI_COMM_WORLD);

//     // Each process prints its received numbers
//     printf("Process %d received:", rank);
//     for (int i = 0; i < elements_per_process; i++) {
//         printf(" %d", recv_nums[i]);
//     }
//     printf("\n");

//     // Cleanup
//     if (rank == 0) {
//         free(data);
//     }
//     free(recv_nums);

//     MPI_Finalize();
//     return 0;
// }


#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // Data on root
    int sendcounts[4] = {2, 3, 2, 3};  // Elements each process gets
    int displs[4] = {0, 2, 5, 7};      // Start index for each process
    int recvbuf[3];  // Maximum possible received elements

    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT,
                 recvbuf, sendcounts[rank], MPI_INT, 
                 0, MPI_COMM_WORLD);

    // Print received data
    printf("Process %d received: ", rank);
    for (int i = 0; i < sendcounts[rank]; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}




// MPI_Scatter(
//     void* send_data,
//     int send_count,
//     MPI_Datatype send_datatype,
//     void* recv_data,
//     int recv_count,
//     MPI_Datatype recv_datatype,
//     int root,
//     MPI_Comm communicator)


// MPI_Scatterv(
//     const void *sendbuf,   // Data to be scattered (on root)
//     const int *sendcounts, // Number of elements each process receives
//     const int *displs,     // Offsets in `sendbuf` where each chunk starts
//     MPI_Datatype sendtype, // Data type of elements in `sendbuf`
//     void *recvbuf,         // Buffer to store received data (on each process)
//     int recvcount,         // Number of elements received (on each process)
//     MPI_Datatype recvtype, // Data type of elements in `recvbuf`
//     int root,              // Rank of the root process
//     MPI_Comm comm          // MPI communicator
// );
