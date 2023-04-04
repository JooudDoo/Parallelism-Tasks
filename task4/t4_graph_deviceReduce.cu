#include <iostream>

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_runtime.h"

#include "sub.cuh" // contains functions for processing arguments and displaying them

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#define at(arr, x, y) (arr[(x)*(n)+(y)])

constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int ITERS_BETWEEN_UPDATE = 400;

void initArrays(double* mainArr, double* main_D, double* sub_D, cmdArgs* args);

__global__ void iterate(double* F, double* Fnew, double* subs, const cmdArgs* args);

int main(int argc, char *argv[]){
    cudaSetDevice(2);
    cmdArgs args = cmdArgs{false, false, 1E-6, (int)1E6, 10, 10}; // create default command line arguments 
    processArgs(argc, argv, &args);
    printSettings(&args);

    double* F_H;
    double* F_D, *Fnew_D;
    size_t size = args.n*args.m*sizeof(double);
    double error = 0;
    int iterationsElapsed = 0;

    cudaMalloc(&F_D, size);
    cudaMalloc(&Fnew_D, size);
    
    F_H = (double*)calloc(sizeof(double), size);

    initArrays(F_H, F_D, Fnew_D, &args);

    // Основной алгоритм здесь


    {
    size_t grid_size = args.n * args.m;

    cmdArgs* args_d;
    cudaMalloc(&args_d, sizeof(cmdArgs));
    cudaMemcpy(args_d, &args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

    double* substractions;
    cudaMalloc(&substractions, size);

    double* error_d;
    cudaMalloc(&error_d, sizeof(double));

    // prepare graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    //prepare reduction
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, substractions, error_d, grid_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    dim3 threadPerBlock = dim3((args.n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK, (args.m + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK);
    dim3 blocksPerGrid = dim3((args.n + ((args.m + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK) - 1) / ((args.m + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK),
            (args.n + ((args.n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK) - 1) / ((args.m + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK));
    
    for(size_t i = 0; i < ITERS_BETWEEN_UPDATE/2; i++){
        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(F_D, Fnew_D, substractions, args_d);
        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(Fnew_D, F_D, substractions, args_d);
    }
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif

    do {

        cudaGraphLaunch(instance, stream);
        cudaDeviceSynchronize();
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, substractions, error_d, grid_size, stream);
        cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);

        iterationsElapsed += ITERS_BETWEEN_UPDATE;
    } while(error > args.eps && iterationsElapsed < args.iterations);

#ifdef NVPROF_
    nvtxRangePop();
#endif

    cudaFree(substractions);
    }


// ----------------------

    std::cout << "Iterations: " << iterationsElapsed << std::endl;
    std::cout << "Error: " << error << std::endl;
    if(args.showResultArr){
        cudaMemcpy(F_H, Fnew_D, size, cudaMemcpyDeviceToHost);
        int n = args.n;
        for(int x = 0; x < args.n; x++){
            for(int y = 0; y < args.m; y++){ 
                std::cout << at(F_H, x, y) << ' ';
            }
            std::cout << std::endl;
        }
    }

    cudaFree(F_D);
    cudaFree(Fnew_D);
    free(F_H);
    return 0;
}

void initArrays(double* mainArr, double* main_D, double* sub_D, cmdArgs* args){
    int n = args->n;
    int m = args->m;
    size_t size = n*m*sizeof(double);

    for(int i = 0; i < n*m && args->initUsingMean; i++){
        mainArr[i] = (LEFT_UP+LEFT_DOWN+RIGHT_UP+RIGHT_DOWN)/4;
    }

    at(mainArr, 0, 0) = LEFT_UP;
    at(mainArr, 0, m-1) = RIGHT_UP;
    at(mainArr, n-1, 0) = LEFT_DOWN;
    at(mainArr, n-1, m-1) = RIGHT_DOWN;
    for(int i = 1; i < n-1; i++){
        at(mainArr,0,i) = (at(mainArr,0,m-1)-at(mainArr,0,0))/(m-1)*i+at(mainArr,0,0);
        at(mainArr,i,0) = (at(mainArr,n-1,0)-at(mainArr,0,0))/(n-1)*i+at(mainArr,0,0);
        at(mainArr,n-1,i) = (at(mainArr,n-1,m-1)-at(mainArr,n-1,0))/(m-1)*i+at(mainArr,n-1,0);
        at(mainArr,i,m-1) = (at(mainArr,n-1,m-1)-at(mainArr,0,m-1))/(m-1)*i+at(mainArr,0,m-1);
    }
    cudaMemcpy(main_D, mainArr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(sub_D, mainArr, size, cudaMemcpyHostToDevice);
}

__global__ void iterate(double* F, double* Fnew, double* subs, const cmdArgs* args){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(j == 0 || i == 0 || i == args->n-1 || j == args->n-1) return; // Dont update borders

    int n = args->n;
    at(Fnew, i, j) = 0.25 * (at(F, i+1, j) + at(F, i-1, j) + at(F, i, j+1) + at(F, i, j-1));
    at(subs, i, j) = fabs(at(Fnew, i, j) - at(F, i, j));
}
