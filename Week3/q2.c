#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	int rank,size,i,M;
	float sum,avg;
	int * arr1;
	int * arr2;
	sum=0;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if(rank==0)
	{
		printf("Enter value of M: \n");
		scanf("%d",&M);
		arr1=(int *)calloc(	size*M,sizeof(M));
		printf("Enter %d values: \n",size*M);
		for(i=0;i<size*M;i++)
		{
			scanf("%d",&arr1[i]);
		}
	}
	MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
	arr2=(int *)calloc(M+1,sizeof(M));

	MPI_Scatter(arr1,M,MPI_INT,arr2,M,MPI_INT,0,MPI_COMM_WORLD);

	for(i=0;i<M;i++)
	{
		sum=sum+arr2[i];
	}

	sum=sum/M;

	printf("\n Average from process %d is %f",rank,sum);

	MPI_Reduce(&sum,&avg,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank==0)
	{
		avg=avg/size;
		printf("\nAverage of all values is %f",avg);
	}
	MPI_Finalize();
	return 0;
}