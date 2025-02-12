#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 

    char ch = 90; 
    printf("Character: '%c'\n", ch); 
    fflush(stdout); 

    MPI_Finalize(); 
    return 0;
}
