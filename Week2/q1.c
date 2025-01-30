#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

void toggle_case(char *word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (isupper(word[i])) {
            word[i] = tolower(word[i]);
        } else if (islower(word[i])) {
            word[i] = toupper(word[i]);
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size, n;
    char word[100]; 
    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a word: ");
        scanf("%s", word);

        n = strlen(word);  

        MPI_Ssend(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Ssend(word, n + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
     
        MPI_Recv(word, n + 1, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
        printf("Received toggled word: %s\n", word);
    }
    else{
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(word, n + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);     
        
        toggle_case(word);

        MPI_Ssend(word, n + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
