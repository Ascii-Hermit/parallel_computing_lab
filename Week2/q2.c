#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
	int rank,size,n;
	MPI_Init(&argc,&argv);
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank == 0){
		printf("Enter Number : ");
		scanf("%d",&n);
		for(int i=1;i<size;i++){
			MPI_Send(&n,1,MPI_INT,i,i,MPI_COMM_WORLD);
		}
	}
	else{
		MPI_Recv(&n,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
		printf("From rank %d received %d \n",rank,n);
	}

	MPI_Finalize();
	return 0;
}