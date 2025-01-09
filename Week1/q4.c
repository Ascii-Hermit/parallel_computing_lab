#include <stdio.h>
#include <mpi.h>

void toggle_case(char* ch, int rank) {
    if (rank % 2 == 0) {
        if (*ch >= 'a' && *ch <= 'z') {
            *ch = *ch - 32;  
        }
    } else {
        if (*ch >= 'A' && *ch <= 'Z') {
            *ch = *ch + 32;  
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    char input_string[] = "HELLO"; 
    int string_length;
    char local_char;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string_length = 5;  
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(input_string, string_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD); 
    int ind = rank % string_length; 
    local_char = input_string[ind];
    toggle_case(&local_char, rank); 

    char result_string[string_length + 1];  
    MPI_Gather(&local_char, 1, MPI_CHAR, result_string, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result_string[string_length] = '\0'; 
        printf("Modified string: %s\n", result_string);
    }

    MPI_Finalize();
    return 0;
}
