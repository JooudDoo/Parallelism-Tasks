#include <iostream>

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#define at(arr, x, y) (arr[(x) * (n) + (y)])


void initArrays(double *mainArr, double *main_D, double *sub_D, cmdArgs *args);

__global__ void iterate(double *F, double *Fnew, const cmdArgs *args);

__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out);

/*
    Функция заполняет массив на хосте - mainArr
    После заполннения на хосте копирует его на main_D и sub_D, расположенные на Device
    Для определение размеров, а также режима заполнения используются аргументы командной строки - cmdArgs
*/
void initArrays(double *mainArr, double *main_D, double *sub_D, cmdArgs *args){
    int n = args->n;
    int m = args->m;
    size_t size = n * m * sizeof(double);

    for (int i = 0; i < n * m && args->initUsingMean; i++){
        mainArr[i] = (LEFT_UP + LEFT_DOWN + RIGHT_UP + RIGHT_DOWN) / 4;
    }

    at(mainArr, 0, 0) = LEFT_UP;
    at(mainArr, 0, m - 1) = RIGHT_UP;
    at(mainArr, n - 1, 0) = LEFT_DOWN;
    at(mainArr, n - 1, m - 1) = RIGHT_DOWN;

    for (int i = 1; i < n - 1; i++) {
        at(mainArr, 0, i) = (at(mainArr, 0, m - 1) - at(mainArr, 0, 0)) / (m - 1) * i + at(mainArr, 0, 0);
        at(mainArr, i, 0) = (at(mainArr, n - 1, 0) - at(mainArr, 0, 0)) / (n - 1) * i + at(mainArr, 0, 0);
        at(mainArr, n - 1, i) = (at(mainArr, n - 1, m - 1) - at(mainArr, n - 1, 0)) / (m - 1) * i + at(mainArr, n - 1, 0);
        at(mainArr, i, m - 1) = (at(mainArr, n - 1, m - 1) - at(mainArr, 0, m - 1)) / (m - 1) * i + at(mainArr, 0, m - 1);
    }
    cudaMemcpy(main_D, mainArr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(sub_D, mainArr, size, cudaMemcpyHostToDevice);
}

__global__ void iterate(double *F, double *Fnew, const cmdArgs *args){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || i == 0 || i == args->n - 1 || j == args->n - 1) return; // Don't update borders

    int n = args->n;
    at(Fnew, i, j) = 0.25 * (at(F, i + 1, j) + at(F, i - 1, j) + at(F, i, j + 1) + at(F, i, j - 1));
}

__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double max_diff = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double diff = abs(in1[i] - in2[i]);
        max_diff = fmax(diff, max_diff);
    }

    double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

    if (threadIdx.x == 0)
    {
    out[blockIdx.x] = block_max_diff;
    }
}