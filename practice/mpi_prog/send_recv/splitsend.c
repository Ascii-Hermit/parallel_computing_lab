// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>
// #include <mpi.h>

// int main(int argc, char* argv[]) {
//     int rank, size, i, len, nvc, tot;
//     nvc = 0; // Number of vowels in local chunk
//     tot = 0;
//     char str1[25]; 
//     char str2[25];  // Buffer for receiving chunks

//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     if (rank == 0) {
//         printf("Enter Required String: \n");
//         fgets(str1, 25, stdin); 

//         len = strlen(str1);
//         if (str1[len - 1] == '\n') {
//             str1[len - 1] = '\0';
//             len--;
//         }
//     }

//     // Broadcast length to all processes
//     MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int chunk_size = len / size;
//     int remainder = len % size;

//     // Send chunks manually
//     if (rank == 0) {
//         for (i = 0; i < size; i++) {
//             int start = i * chunk_size;
//             int count = chunk_size;

//             // Last process gets the remainder characters too
//             if (i == size - 1) {
//                 count += remainder;
//             }

//             MPI_Send(&str1[start], count, MPI_CHAR, i, 0, MPI_COMM_WORLD);
//         }
        
//         // Rank 0 processes its own chunk (excluding remainder)
//         strncpy(str2, str1, chunk_size);
//         str2[chunk_size] = '\0';
//     } else {
//         int recv_size = chunk_size;

//         // Last process receives additional remainder characters
//         if (rank == size - 1) {
//             recv_size += remainder;
//         }

//         MPI_Recv(str2, recv_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         str2[recv_size] = '\0';
//         printf("Process %d got %s\n",rank,str2);
//     }

//     // Count vowels in received chunk
//     for (i = 0; i < strlen(str2); i++) {
//         char ch = str2[i];
//         if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
//             ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') {
//             nvc++;
//         }
//     }

//     // Send vowel count back to rank 0
//     if (rank == 0) {
//         tot = nvc;
//         for (i = 1; i < size; i++) {
//             int temp;
//             MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             tot += temp;
//         }
//         printf("Total number of vowels: %d\n", tot);
//     } else {
//         MPI_Send(&nvc, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }

//     MPI_Finalize();
//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, size, num_vow,len;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    char str1[25];
    char str2[25];

    if(rank == 0){
        printf("Enter you string: \n");
        fgets(str1,25,stdin); // remember this

        len = strlen(str1);
        if(str1[len-1] == '\n'){
            str1[len-1] = '\0';
            len--;
        }
    }

    MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);

    int chunk = len/size;
    int remainder = len%size;

    if(rank == 0){
        for(int i = 1;i<size;i++){
            int chunk_len = chunk;
            int start = i*chunk;
            if(i== size-1){
                chunk_len+=remainder;
            }
            MPI_Send(&str1[start],chunk_len,MPI_CHAR,i,0,MPI_COMM_WORLD);
        }
        strncpy(str2, str1, chunk);
        str2[chunk] = '\0';
    }   
    else{
        int chunk_len = chunk;
        if(rank == size-1){
            chunk_len+=remainder;
        }
        MPI_Recv(&str2,chunk_len,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        str2[chunk_len] = '\0';
        
    }
    for(int i = 0;i<strlen(str2);i++){
            char ch = str2[i];
            if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
            ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') {
                num_vow++;
            }
        }
    if(rank == 0){
        int tot = num_vow;
        for(int i = 1;i<size;i++){
            MPI_Recv(&num_vow,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            tot+=num_vow;    
        }
        printf("The total vowels are %d",tot);
    }
    else{
        MPI_Send(&num_vow,1,MPI_INT,0,0,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}





