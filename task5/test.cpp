#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]){

    int rank;
    int size;
    int cycles = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("%d/%d\n", rank, size);


    int ack;

    if(rank == 0){
        MPI_Status status;
        printf("rank zero (%d)\n", cycles);
        ack = 1;
        while(cycles > 0){
            MPI_Send(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&ack, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, &status);
            printf("end cycle %d\n", cycles);
            cycles -= 1;
        }
        printf("recived end: %d %d\n", ack, rank);
    }
    else if (rank != 0){
        MPI_Status status;
        while(cycles> 0){
            MPI_Recv(&ack, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
            printf("recived %d - %d\n", rank, ack);
            ack += 1;
            int sentTo = rank+1!=size?rank+1:0;,
            MPI_Send(&ack, 1, MPI_INT, sentTo, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}