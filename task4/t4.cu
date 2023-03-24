#include <iostream>

#include "cuda_runtime.h"

#include "sub.cuh" // contains functions for processing arguments and displaying them

#define at(arr, x, y) (arr[(x)*(n)+(y)])

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

void initArrays(double* mainArr, double* main_D, double* sub_D, cmdArgs* args);

__global__ void solve(double* F, double* Fnew, cmdArgs* args, double* error, int* iterationsElapsed);

__global__ void iterate(double* F, double* Fnew, cmdArgs* args);

int main(int argc, char *argv[]){
    cmdArgs args = cmdArgs{false, false, 1E-6, (int)1E6, 10, 10}; // create default command line arguments 
    processArgs(argc, argv, &args);
    printSettings(&args);

    double* F_H;
    double* F_D, *Fnew_D;
    size_t size = args.n*args.m*sizeof(double);

    cudaMalloc(&F_D, size);
    cudaMalloc(&Fnew_D, size);
    F_H = (double*)calloc(sizeof(double), size);

    initArrays(F_H, F_D, Fnew_D, &args);

// Основной алгоритм здесь

    int iterationsElapsed = 0;
    double error = 0;
    {
        cmdArgs* args_d;
        double* error_d;
        int* iterations_d;
        cudaMalloc(&args_d, sizeof(cmdArgs));
        cudaMalloc(&error_d, sizeof(double));
        cudaMalloc(&iterations_d, sizeof(int));

        cudaMemcpy(args_d, &args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

        solve<<<1, 1>>>(F_D, Fnew_D, args_d, error_d, iterations_d);

        cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&iterationsElapsed, iterations_d, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(error_d);
        cudaFree(args_d);
        cudaFree(iterations_d);
    }

// ----------------------

    std::cout << "Iterations: " << iterationsElapsed << std::endl;
    std::cout << "Error: " << error << std::endl;
    if(args.showResultArr){
        cudaMemcpy(F_H, F_D, size, cudaMemcpyDeviceToHost);
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

__global__ void iterate(double* F, double* Fnew, cmdArgs* args){
    if(blockIdx.x == 0 || threadIdx.x == 0) return; // Dont update borders
    int n = args->n;
    at(Fnew, blockIdx.x, threadIdx.x) = 0.25 * (at(F, blockIdx.x+1, threadIdx.x) + at(F, blockIdx.x-1, threadIdx.x) + at(F, blockIdx.x, threadIdx.x+1) + at(F, blockIdx.x, threadIdx.x-1));
}

__global__ void solve(double* F, double* Fnew, cmdArgs* args, double* error, int* iterationsElapsed){

    *error = 1;
    do {

        iterate<<<args->n-1, args->m-1>>>(F, Fnew, args);

        double* swap = F;
        F = Fnew;
        Fnew = swap;

        (*iterationsElapsed)++;
    } while(*error > args->eps && *iterationsElapsed < args->iterations);
}