
#include <mpi.h>

#include "cuda_runtime.h"

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
constexpr int THREADS_PER_BLOCK_REDUCE = 256;

#include "sub.cuh"
#include "cudaFuncs.cuh"


int main(int argc, char *argv[]){

// ------------------
// Подготовка к работе
// ------------------

    // Инициализируем MPI
    int rank, process_count;
    MPI_Init(&argc, &argv);

    // Определяем сколько процессов внутри глоабльного коммуникатора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    cudaSetDevice(rank % 4);

    cmdArgs args = cmdArgs{false, false, 1E-6, (int)1E6, 10, 10};
    processArgs(argc, argv, &args);

    if(rank == 0){
        printSettings(&args);
    }

    // // Создаем новые аргументы командной стороки
    // // Т.к. каждый поток обрабатывает кусок сетки
    // // И мы ведем обмен граничными условиями, то
    // // n * m - размер становится:
    // //      ELEMS_BY_PROCESS / m * m
    // // А еще нужно учесть то что у крайних граничными условиями нужно обмениватся только с одной стороны, получаем:
    // // (ELEMS_BY_PRORCESS / m + 1 + ((rank % (process_count-1)) != 0)
    // cmdArgs args_by_proc {args};
    // args_by_proc.n = ELEMENTS_BY_PROCESS / args.m + 1 + ((rank % (process_count-1)) != 0);

    int TOTAL_GRID_SIZE = args.m * args.n;
    int ELEMENTS_BY_PROCESS = TOTAL_GRID_SIZE / process_count;

    // ====== Создаем указатели на массивы для GPU/CPU ======
    double *F_H;
    double *F_D, *Fnew_D;
    double *F_H_full = nullptr; // Указатель для хранения всего массива (используется в rank = 0)
    size_t array_size_bytes = ELEMENTS_BY_PROCESS * sizeof(double);

    // ====== Переменные для хранения рантайм значений ======
    double error = 0;
    int iterationsElapsed = 0;

    // ====== Выделяем память для GPU ======
    cudaMalloc(&F_D, array_size_bytes);
    cudaMalloc(&Fnew_D, array_size_bytes);

    F_H = (double*)malloc(array_size_bytes);

// ------------------
// Иницилизируем массив в 0 процессе и отправляем всем остальным процессам их части
// ------------------
    { 
        int n = args.n;
        int m = args.m;

        // Заполняем полный массив в 0 процессе 
        if(rank == 0){
            
            F_H_full = (double*)calloc(n*m, sizeof(double));

            for (int i = 0; i < n * m && args.initUsingMean; i++){
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

            
        }

        MPI_Scatter(
            F_H_full,
            ELEMENTS_BY_PROCESS,
            MPI_DOUBLE,
            F_H,
            ELEMENTS_BY_PROCESS,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

    }

// ------------------
// Основной цикл программы
// !TODO описание работы основного цикла
// ------------------
    {
        bool isWorking = true;

        size_t grid_size = ELEMENTS_BY_PROCESS;

        int num_blocks_reduce = (grid_size + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE;

        double *error_reduction;
        cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);
        double *error_d;
        cudaMalloc(&error_d, sizeof(double));

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cmdArgs proc_args = {args};
        proc_args.n = ELEMENTS_BY_PROCESS / args.m;

        cmdArgs *args_d;
        cudaMalloc(&args_d, sizeof(cmdArgs));
        cudaMemcpy(args_d, &proc_args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

        int threadPerBlock = 32;
        int blocksPerGrid  = proc_args.n;


        while(isWorking){
            iterate<<<blocksPerGrid, threadPerBlock>>>(F_D, Fnew_D, args_d);

            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F_D, Fnew_D, grid_size, error_reduction);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce, stream);

            iterate<<<blocksPerGrid, threadPerBlock>>>(Fnew_D, F_D, args_d);

            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F_D, Fnew_D, grid_size, error_reduction);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce, stream);

            cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

// ------------------
// Собираем итоговый массив на нулевой процесс и выводим его
// Выполняем это только если стоит соответственный флаг 
// ------------------
    if(args.showResultArr){
        MPI_Gather(
            F_H,
            ELEMENTS_BY_PROCESS,
            MPI_DOUBLE,
            F_H_full,
            ELEMENTS_BY_PROCESS,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );
        
        if(rank == 0){
            int n = args.n;
            for (int x = 0; x < args.n; x++) {
                for (int y = 0; y < args.m; y++) {
                    std::cout << at(F_H_full, x, y) << ' ';
                }
                std::cout << std::endl;
            }
        }
    }



    if(F_H_full) free(F_H_full);
    cudaFree(F_D);
    cudaFree(Fnew_D);
    free(F_H);

    MPI_Finalize();
    return 0;
}