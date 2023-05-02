#include <iostream>

#include <mpi.h>
#include <nccl.h>

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#define at(arr, x, y) (arr[(x) * (n) + (y)])

__global__ void iterate(double *F, double *Fnew, const cmdArgs *args);

__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out);

/*
* Функция производит обмен граничными условиями между процессами
* Обмен происходит по принципу того что необходимо отправить процессу выше/ниже данные (при наличии такового)
* Для процесса ниже [т.е. его rank+1 от нашего (ниже т.к. обрабатывает часть сетки снизу)]
*   Происходит отправка предпоследней строчки части сетки текущие процесса (она становится его границей)
* Для процесса выше [т.е. его rank-1 от нашего (выше т.к. обрабатывает часть сетки сверху)]
*   Происходит отправка второй строчки
* Соотвественно принимаем от процессов сверху/снизу данные для наших границ

'#' - граничные данные части
'$' - значение которое изменяется в ходе расчетов

Тогда на сетке 16x16 для 4 потоков картина такая:

        rank = 0
0 :  ################
1 :  #$$$$$$$$$$$$$$#
2 :  #$$$$$$$$$$$$$$#
3 :  #$$$$$$$$$$$$$$#
4 :  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
5 :  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

         rank = 1
4 :  ################ -- эта строчка получена от верхнего процесса
5 :  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
6 :  #$$$$$$$$$$$$$$#
7 :  #$$$$$$$$$$$$$$#
8 :  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
9 :  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

        rank = 2
8 :  ################ -- эта строчка получена от верхнего процесса
9 :  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
10:  #$$$$$$$$$$$$$$#
11:  #$$$$$$$$$$$$$$#
12:  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
13:  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

        rank = 3
12:  ################ -- эта строчка получена от верхнего процесса
13:  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
14:  #$$$$$$$$$$$$$$#
15:  ################

*/
void transfer_data(const int rank, const int ranks_count, double* F_from, double* F_to, cmdArgs& local_args){

    if(rank != 0){
        cudaMemcpy(F_from + local_args.m, F_to + local_args.m, local_args.m * sizeof(double), cudaMemcpyDeviceToHost);
        MPI_Request rq;
        // отправляем указатель на вторую строку процессу, работующему сверху
        MPI_Isend(
            F_from + local_args.m,
            local_args.m,
            MPI_DOUBLE,
            rank-1,
            rank-1,
            MPI_COMM_WORLD,
            &rq
        );
    }

    // отправляем свою вторую строку вниз 
    if(rank != ranks_count-1){
        MPI_Request rq;
        cudaMemcpy(F_from + local_args.m*(local_args.n-2), F_to + local_args.m*(local_args.n-2), local_args.m * sizeof(double), cudaMemcpyDeviceToHost);
        MPI_Isend(
            F_from + local_args.m*(local_args.n-2),
            local_args.m,
            MPI_DOUBLE,
            rank+1,
            rank+1,
            MPI_COMM_WORLD,
            &rq
        );
    }

    // принимаем строку от верхнего
    if(rank != 0){
        MPI_Status status;
        MPI_Recv(F_from, local_args.m, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpy(F_to, F_from, local_args.m * sizeof(double), cudaMemcpyHostToDevice);

    }
    // принимаем строку от нижнего
    if(rank != ranks_count - 1){
        MPI_Status status;
        MPI_Recv(F_from+(local_args.m * (local_args.n-1)), local_args.m, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpy(F_to+(local_args.m * (local_args.n-1)), F_from+(local_args.m * (local_args.n-1)), local_args.m * sizeof(double), cudaMemcpyHostToDevice);
    }
}


#ifdef NCCL_INC
/*
* Функция производит обмен граничными условиями между процессами
* Обмен происходит по принципу того что необходимо отправить процессу выше/ниже данные (при наличии такового)
* Для процесса ниже [т.е. его rank+1 от нашего (ниже т.к. обрабатывает часть сетки снизу)]
*   Происходит отправка предпоследней строчки части сетки текущие процесса (она становится его границей)
* Для процесса выше [т.е. его rank-1 от нашего (выше т.к. обрабатывает часть сетки сверху)]
*   Происходит отправка второй строчки
* Соотвественно принимаем от процессов сверху/снизу данные для наших границ

'#' - граничные данные части
'$' - значение которое изменяется в ходе расчетов

Тогда на сетке 16x16 для 4 потоков картина такая:

        rank = 0
0 :  ################
1 :  #$$$$$$$$$$$$$$#
2 :  #$$$$$$$$$$$$$$#
3 :  #$$$$$$$$$$$$$$#
4 :  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
5 :  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

         rank = 1
4 :  ################ -- эта строчка получена от верхнего процесса
5 :  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
6 :  #$$$$$$$$$$$$$$#
7 :  #$$$$$$$$$$$$$$#
8 :  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
9 :  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

        rank = 2
8 :  ################ -- эта строчка получена от верхнего процесса
9 :  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
10:  #$$$$$$$$$$$$$$#
11:  #$$$$$$$$$$$$$$#
12:  #$$$$$$$$$$$$$$# -- эта строка уйдет нижнему процессу после шага итерации
13:  ################ -- эта строчка будет получена от нижнего процесса после шага итерации

        rank = 3
12:  ################ -- эта строчка получена от верхнего процесса
13:  #$$$$$$$$$$$$$$# -- эта строчка отправлена верхнему процессу
14:  #$$$$$$$$$$$$$$#
15:  ################

*/
void transfer_data_nccl(const int rank, const int ranks_count, double* F_from, double* F_to, cmdArgs& local_args, ncclComm_t comm){
    ncclGroupStart();
    if(rank != 0){
        ncclSend(
            F_from + local_args.m,
            local_args.m,
            ncclDouble,
            rank-1,
            comm,
            0
        );
    }
    if(rank != ranks_count-1){
        ncclSend(
            F_from + local_args.m*(local_args.n-2),
            local_args.m,
            ncclDouble,
            rank+1,
            comm,
            0
        );
    }

    if(rank != 0){
        ncclRecv(
            F_to,
            local_args.m,
            ncclDouble,
            rank-1,
            comm,
            0
        );
    }

    if(rank != ranks_count - 1){
        ncclRecv(
            F_to+(local_args.m * (local_args.n-1)),
            local_args.m,
            ncclDouble,
            rank+1,
            comm,
            0
        );
    }
    ncclGroupEnd();
}
#endif

#ifdef NCCL_INC
void transfer_data_nccl_cuda(const int* rank, const int* ranks_count, double* F_from, double* F_to, cmdArgs* local_args, ncclComm_t* comm){
    ncclGroupStart();
    if((*rank) != 0){
        ncclSend(
            F_from + local_args->m,
            local_args->m,
            ncclDouble,
            (*rank)-1,
            *comm,
            0
        );
    }
    if((*rank) != (*ranks_count)-1){
        ncclSend(
            F_from + local_args->m*(local_args->n-2),
            local_args->m,
            ncclDouble,
            (*rank)+1,
            *comm,
            0
        );
    }

    if((*rank) != 0){
        ncclRecv(
            F_to,
            local_args->m,
            ncclDouble,
            (*rank)-1,
            *comm,
            0
        );
    }

    if((*rank) != (*ranks_count) - 1){
        ncclRecv(
            F_to+(local_args->m * (local_args->n-1)),
            local_args->m,
            ncclDouble,
            (*rank)+1,
            *comm,
            0
        );
    }
    ncclGroupEnd();
}
#endif

__global__ void iterate(double *F, double *Fnew, const cmdArgs *args){

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0 || i == 0 || i >= args->n - 1 || j >= args->m - 1) return; // Don't update borders

    int n = args->m;
    at(Fnew, i, j) = 0.25 * (at(F, i + 1, j) + at(F, i - 1, j) + at(F, i, j + 1) + at(F, i, j - 1));
}


__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK_REDUCE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double max_diff = 0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double diff = fabs(in1[i] - in2[i]);
    max_diff = fmax(diff, max_diff);

    double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());
    
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = block_max_diff;
    }
}

// __global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
//     typedef cub::BlockReduce<double, 256> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     double max_diff = 0;

//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
//         double diff = abs(in1[i] - in2[i]);
//         max_diff = fmax(diff, max_diff);
//     }

//     double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

//     if (threadIdx.x == 0)
//     {
//     out[blockIdx.x] = block_max_diff;
//     }
// }

// __global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
//     typedef cub::BlockReduce<double, THREADS_PER_BLOCK_REDUCE> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     double max_diff = 0;

//     int i = threadIdx.x;

//     while (i < n)
//     {
//         double diff = fabs(in1[i] - in2[i]);
//         max_diff = fmax(diff, max_diff);

//         i += blockDim.x;
//     }

//     double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

//     if (threadIdx.x == 0)
//     {
//         out[blockIdx.x] = block_max_diff;
//     }
// }