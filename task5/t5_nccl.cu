#include <mpi.h>
#include <nccl.h>

#include "cuda_runtime.h"

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#ifdef DEBUG
#define DEBUG_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG_PRINTF(line, a...) 0
#endif

#ifdef DEBUG1
#define DEBUG1_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG1_PRINTF(line, a...) 0
#endif

#define NCCL_INC

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
constexpr int THREADS_PER_BLOCK_REDUCE = 256;

constexpr int ITERS_BETWEEN_UPDATE = 400;

#include "sub.cuh"
#include "cudaFuncs.cuh"


int main(int argc, char *argv[]) {

// ------------------
// Подготовка к работе
// ------------------

    // ====== Инициализируем MPI ======
    int rank, ranks_count;
    MPI_Init(&argc, &argv);

    // ====== Определяем сколько процессов внутри глоабльного коммуникатора ======
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);

    // ====== Каждый процесс выбирает свою видеокарту ======
    cudaSetDevice(rank);

    // ====== Парсинг аргументов командной строки ======
    cmdArgs global_args = cmdArgs{false, false, 1E-6, (int)1E6, 10, 10};
    processArgs(argc, argv, &global_args);

    if(rank == 0){
        printSettings(&global_args);
    }

    // ====== Расчет элементов для каждого процесса ======
    int TOTAL_GRID_SIZE = global_args.m * global_args.n; 

    cmdArgs local_args{global_args};
    local_args.n =  TOTAL_GRID_SIZE / ranks_count / global_args.m + 2 * (rank != ranks_count - 1);
    
    int ELEMENTS_BY_PROCESS = local_args.n * local_args.m;

    // ====== Создание указателей на массивы ======
    double *F_H_full = nullptr; // Указатель для хранения всего массива (используется в rank = 0)
    double *error_array = nullptr; // Указатель для хранения массива ошибок полученных с остальных процессов на нулевой (используется в rank 0)
    double *F_H;
    double *F_D, *Fnew_D;
    size_t array_size_bytes = ELEMENTS_BY_PROCESS * sizeof(double);

    // ====== Выделяем память для GPU/CPU ======
    cudaMalloc(&F_D, array_size_bytes);
    cudaMalloc(&Fnew_D, array_size_bytes);

    cudaMallocHost(&F_H, array_size_bytes);

    if(rank == 0){
        error_array = (double*)malloc(sizeof(double) * ranks_count);
    }

// ------------------
// Создаем коммуникатор для nccl
// ------------------
    ncclUniqueId nccl_id;
    if (rank == 0) 
        ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, ranks_count, nccl_id, rank);

// ------------------
// Иницилизируем массив в 0 процессе и отправляем всем остальным процессам их части
// Каждый процесс обрабатывает local_args.n - строк
// Это значение зависит от того какой участок обрабатывает наш процесс
// В итоге получаем что каждый процесс обрабатывает global_args.n / 4 + 2 строк
// Кроме последнего, он обрабатывает только global_args.n / 4 строк
// Это происходит из-за того, что ему нет необходимости поддерживать граничные значения с нижним блоком (он является самым нижним)
// ------------------
{ 
    int n = global_args.n;
    int m = global_args.m;

    // Заполняем полный массив в 0 процессе 
    if(rank == 0){
        
        F_H_full = (double*)calloc(n*m, sizeof(double));

        for (int i = 0; i < n * m && global_args.initUsingMean; i++){
            F_H_full[i] = (LEFT_UP + LEFT_DOWN + RIGHT_UP + RIGHT_DOWN) / 4;
        }

        at(F_H_full, 0, 0) = LEFT_UP;
        at(F_H_full, 0, m - 1) = RIGHT_UP;
        at(F_H_full, n - 1, 0) = LEFT_DOWN;
        at(F_H_full, n - 1, m - 1) = RIGHT_DOWN;

        for (int i = 1; i < n - 1; i++) {
            at(F_H_full, 0, i) = (at(F_H_full, 0, m - 1) - at(F_H_full, 0, 0)) / (m - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, i, 0) = (at(F_H_full, n - 1, 0) - at(F_H_full, 0, 0)) / (n - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, n - 1, i) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, n - 1, 0)) / (m - 1) * i + at(F_H_full, n - 1, 0);
            at(F_H_full, i, m - 1) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, 0, m - 1)) / (m - 1) * i + at(F_H_full, 0, m - 1);
        }

        int data_start = 0;
        int data_lenght = 0;

    // ------------------
    // Отправляем необходимые части всем процессам, включая самого себя
    // ------------------
        for(size_t target = 0; target < ranks_count; target++){
            MPI_Request req;
            data_lenght = ELEMENTS_BY_PROCESS - 2 * local_args.m * (target == (ranks_count - 1) && ranks_count != 1);
            DEBUG_PRINTF("Sended to %d elems: %d from: %d\n", target, data_lenght, data_start);
            MPI_Isend(
                F_H_full + data_start,
                data_lenght,
                MPI_DOUBLE,
                target,
                0,
                MPI_COMM_WORLD,
                &req
            );
            
            data_start += data_lenght - local_args.m * 2;
            
        }
        
    }

// ------------------
// Ждём получения обрабатываемой части от 0 процесса
// ------------------
    MPI_Status status;
    MPI_Recv(
        F_H,
        ELEMENTS_BY_PROCESS,
        MPI_DOUBLE,
        0,
        0,
        MPI_COMM_WORLD,
        &status
    );

    cudaMemcpy(F_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Fnew_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);

}

    double error = 1;
    int iterationsElapsed = 0;


// ------------------
// Основной цикл работы программы
// Иницилизация ->
//  =========== ЦИКЛ ============
//      Проход по своему участку
//          Обмен граничными условиями
//      Проход по своему участку
//          Обмен граничными условиями
//      Расчет ошибки
//          Сбор ошибок с каждого процесса и вычисление общей
//          Отправка общей ошибки каждому процессу (Маркер того что основной процесс обработал все их отправленные данные)
// ------------------
{

    cmdArgs *args_d;
    cudaMalloc(&args_d, sizeof(cmdArgs));
    cudaMemcpy(args_d, &local_args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

    int num_blocks_reduce = (ELEMENTS_BY_PROCESS + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE;

    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);
    double *error_d;
    cudaMalloc(&error_d, sizeof(double));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


    dim3 threadPerBlock {(local_args.n + MAXIMUM_THREADS_PER_BLOCK - 1)/MAXIMUM_THREADS_PER_BLOCK,
                         (local_args.m + MAXIMUM_THREADS_PER_BLOCK - 1)/MAXIMUM_THREADS_PER_BLOCK}; // ПЕРЕОСМЫСЛИТЬ
    if(threadPerBlock.x > 0){
        threadPerBlock.x = 32;
    }
    if(threadPerBlock.y > 0){
        threadPerBlock.y = 32;
    } 
    dim3 blocksPerGrid {(local_args.n + threadPerBlock.x - 1)/threadPerBlock.x,
                        (local_args.m + threadPerBlock.y - 1)/threadPerBlock.y}; // ПЕРЕОСМЫСЛИТЬ

    DEBUG_PRINTF("%d: %d %d %d %d\n", rank, threadPerBlock.x, threadPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

    MPI_Barrier(MPI_COMM_WORLD);

    do {
        iterate<<<blocksPerGrid, threadPerBlock>>>(F_D, Fnew_D, args_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data_nccl(rank, ranks_count, Fnew_D, Fnew_D, local_args, comm);

        iterate<<<blocksPerGrid, threadPerBlock>>>(Fnew_D, F_D, args_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data_nccl(rank, ranks_count, F_D, F_D, local_args, comm);

        iterationsElapsed += 2;
        if(iterationsElapsed % ITERS_BETWEEN_UPDATE == 0){
            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F_D, Fnew_D, ELEMENTS_BY_PROCESS, error_reduction);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
            cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);

// ------------------
// Сборка ошибок с каждого процесса и обработка их на 0 потоке (Procces reduction)
// ------------------
            {
                MPI_Gather(&error, 1, MPI_DOUBLE, error_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if(rank == 0){
                    error = 0;
                    for(int err_id = 0; err_id < ranks_count; err_id++){
                        error = max(error, error_array[err_id]);
                    }
                    DEBUG1_PRINTF("iters: %d error: %lf\n", iterationsElapsed, error);
                }
                MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    } while(error > global_args.eps && iterationsElapsed < global_args.iterations);

}


    if(global_args.showResultArr){
// ------------------
// Отправка финального массива на нулевой процесс
// ------------------
        cudaMemcpy(F_H, F_D, array_size_bytes, cudaMemcpyDeviceToHost);
        MPI_Request req;
        MPI_Isend(F_H + local_args.m, ELEMENTS_BY_PROCESS - (local_args.m * 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);

// ------------------
// Отображение финального массива с нулевого процесса
// ------------------
        if(rank == 0){
            int array_offset = local_args.m;
            for(int target = 0; target < ranks_count; target++){
                MPI_Status status;
                int recive_size = ELEMENTS_BY_PROCESS - 2 * local_args.m - 2 * local_args.m * (target == (ranks_count - 1));
                MPI_Recv(F_H_full + array_offset, recive_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &status);
                array_offset += recive_size;
            }

            std::cout << rank << " ---\n";
            for (int x = 0; x < global_args.n; x++) {
                    int n = global_args.m;
                    for (int y = 0; y < global_args.m; y++) {
                        std::cout << at(F_H_full, x, y) << ' ';
                    }
                    std::cout << std::endl;
                }
            std::cout << std::endl;
        }
    }

    if(rank == 0){
        std::cout << "Iterations: " << iterationsElapsed << std::endl;
        std::cout << "Error: " << error << std::endl;
    }

    if(F_H_full) free(F_H_full);
    if(error_array) free(error_array);
    cudaFree(F_D);
    cudaFree(Fnew_D);
    cudaFree(F_H);

    MPI_Finalize();
    return 0;
}