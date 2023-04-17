
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
    cudaSetDevice(rank % 4);

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
    double *F_H;
    double *F_D, *Fnew_D;
    size_t array_size_bytes = ELEMENTS_BY_PROCESS * sizeof(double);

    // ====== Выделяем память для GPU/CPU ======
    cudaMalloc(&F_D, array_size_bytes);
    cudaMalloc(&Fnew_D, array_size_bytes);

    F_H = (double*)malloc(array_size_bytes);

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
            data_lenght = ELEMENTS_BY_PROCESS - 2 * local_args.m * (target == (ranks_count - 1));
            std::cout << "Sended to " << target << " elems:" << data_lenght << " from: " << data_start << std::endl;
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

}

    double error = 0;
    int iterationsElapsed = 0;


// ------------------
// Основной цикл работы программы
// Иницилизация ->
//  =========== ЦИКЛ ============
//      Проход по своему участку
//      Обмен граничными условиями
//      Проход по своему участку
//      Обмен граничными условиями
//      Расчет ошибки
//      Сбор ошибок с каждого процесса и вычисление общей
//      Отправка общей ошибки каждому процессу (Маркер того что основной процесс обработал все их отправленные данные)
// ------------------
{

    cmdArgs *args_d;
    cudaMalloc(&args_d, sizeof(cmdArgs));
    cudaMemcpy(args_d, &local_args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

    int num_blocks_reduce = (ELEMENTS_BY_PROCESS + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE;

    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);
    double *error_d;
    cudaMalloc(&error_d, sizeof(double))

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int threadPerBlock = 32; // ПЕРЕОСМЫСЛИТЬ
    int blocksPerGrid  = local_args.n + threadPerBlock.x - 1; // ПЕРЕОСМЫСЛИТЬ

    do {
        iterate<<<blocksPerGrid, threadPerBlock>>>(F_D, Fnew_D, args_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ

        iterate<<<blocksPerGrid, threadPerBlock>>>(Fnew_D, F_D, args_d);

        // ОБМЕн ГРАНИЧНЫМИ УСЛОВИЯМИ

        block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F_D, Fnew_D, grid_size, error_reduction);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);

        cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);

        // СБОР ОШИБОК СО ВСЕХ ПРОЦЕССОВ
        // ОТПРАВКА ОБЩЕЙ ОШИБКИ С ОСНОВНОГО ПРОЦЕССА НА ОСТАЛЬНЫЕ

        iterationsElapsed += 2;
    } while(error > args.eps && iterationsElapsed < args.iterations)

}

    std::cout << rank << " ---\n";
    for (int x = 0; x < local_args.n; x++) {
            int n = local_args.m;
            for (int y = 0; y < local_args.m; y++) {
                std::cout << at(F_H, x, y) << ' ';
            }
            std::cout << std::endl;
        }
    std::cout << std::endl;

    if(F_H_full) free(F_H_full);
    cudaFree(F_D);
    cudaFree(Fnew_D);
    free(F_H);

    MPI_Finalize();
    return 0;
}