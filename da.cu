#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "mpi.h"

#define UPDATE 250
#define THREADS_MAX 1024
#define THREAD (size < THREADS_MAX ? size : THREADS_MAX)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Функция изменения матрицы
__global__ void iterate(double* A, double* A_new, size_t size_x, size_t size_y) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	
    if ((i == 0) || (j == 0) || (j == size_y - 1) || (i == size_x - 1)) return; // Don't update borders
    A_new[j * size_x + i] = 0.25 * (A[j * size_x + i - 1] + A[(j - 1) * size_x + i] + A[(j + 1) * size_x + i] + A[j * size_x + i + 1]);	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Функция разницы матриц
__global__ void subtraction(double* A, double* A_new, double* A_err, size_t size_x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
	A_err[j * size_x + j] = A[j * size_x + j] - A_new[j * size_x + j];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Значения по умодчанию
double eps = 1E-6;
int size = 512;
int iter_max = 1E6;

int main(int argc, char** argv) {
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Получение значений из командной строки
    for(int arg = 0; arg < argc; arg++){ 
        std::stringstream buffer;
        if(strcmp(argv[arg], "-error") == 0){
            buffer << argv[arg+1];
            buffer >> eps;
        }
        else if(strcmp(argv[arg], "-iter") == 0){
            buffer << argv[arg+1];
            buffer >> iter_max;
        }
        else if(strcmp(argv[arg], "-size") == 0){
            buffer << argv[arg+1];
            buffer >> size;
        }
    }

	size_t totalSize = size * size;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Выбор видеокарт
    int DEVICE, COUNT_DEVICE;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &DEVICE);
    MPI_Comm_size(MPI_COMM_WORLD, &COUNT_DEVICE);

    cudaSetDevice(DEVICE);

    if (DEVICE == 0)
        std::cout << "Settings: " << "\n\tMin error: " << eps << "\n\tMax iteration: " << iter_max << "\n\tSize: " << size << "x" << size << std::endl;

	if (DEVICE!=0)
        cudaDeviceEnablePeerAccess(DEVICE - 1, 0);
    if (DEVICE!=COUNT_DEVICE-1)
        cudaDeviceEnablePeerAccess(DEVICE + 1, 0);

	size_t size_y = size / COUNT_DEVICE + 1;
    if (DEVICE != COUNT_DEVICE - 1 && DEVICE != 0) size_y += 1;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Выделения памяти
	double *A, *A_Device, *A_new_Device, *A_error_Device, *deviceError, *tempStorage = NULL;
    size_t tempStorageSize = 0;

    cudaMallocHost(&A, sizeof(double) * totalSize);

    for (int j = 0; j < size; j++)  {
        A[j] = 10.0 + j * 10.0 / (size - 1);
        A[j * size] = 10.0 + j * 10.0 / (size - 1);
        A[size - 1 + j * size] = 20.0 + j * 10.0 / (size - 1);
        A[size * (size - 1) + j] = 20.0 + j * 10.0 / (size - 1);
    }

    dim3 threads(THREAD);
    dim3 blocks(size/THREAD, size_y);

    printf("%d: %d %d %d %d\n", DEVICE, threads.x, threads.y, blocks.x, blocks.y);

	cudaMalloc(&A_Device, sizeof(double) * size * size_y);
	cudaMalloc(&A_new_Device, sizeof(double) * size * size_y);
	cudaMalloc(&A_error_Device, sizeof(double) * size * size_y);
	cudaMalloc(&deviceError, sizeof(double));


	size_t offset = (DEVICE != 0) ? size : 0;
 	cudaMemcpy(A_Device, A + (size * size * DEVICE / COUNT_DEVICE) - offset, sizeof(double) * size * size_y, cudaMemcpyHostToDevice);
	cudaMemcpy(A_new_Device, A + (size * size * DEVICE / COUNT_DEVICE) - offset, sizeof(double) * size * size_y, cudaMemcpyHostToDevice);

	cub::DeviceReduce::Max(tempStorage, tempStorageSize, A_error_Device, deviceError, size * size_y);
	cudaMalloc(&tempStorage, tempStorageSize);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Основной цикл
    size_t iter = 0;
    double error = 1.0;
	clock_t begin = clock();
	while((iter < iter_max) && error > eps)	{
		iterate<<<blocks, threads>>>(A_Device, A_new_Device, size, size_y);
		iter++;
		// Расчитываем ошибку каждую сотую итерацию
		if (iter % UPDATE == 0) {
            subtraction<<<blocks, threads>>>(A_new_Device, A_Device, A_error_Device, size);
			cub::DeviceReduce::Max(tempStorage, tempStorageSize, A_error_Device, deviceError, size * size_y);
			cudaMemcpyAsync(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);

			// Находим максимальную ошибку среди всех и передаём её всем процессам
			MPI_Allreduce((void*)&error,(void*)&error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		}

		if (DEVICE != 0)                // Обмен верхней границей
            MPI_Sendrecv(A_new_Device + size + 1, size - 2, MPI_DOUBLE, DEVICE - 1, 0, A_new_Device + 1, size - 2, MPI_DOUBLE, DEVICE - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
        if (DEVICE != COUNT_DEVICE - 1) // Обмен нижней границей
            MPI_Sendrecv(A_new_Device + (size_y - 2) * size + 1, size - 2, MPI_DOUBLE, DEVICE + 1, 0, A_new_Device + (size_y - 1) * size + 1, size - 2, MPI_DOUBLE, DEVICE + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		std::swap(A_Device, A_new_Device);
	}

	clock_t end = clock();
	if (DEVICE == 0) {
		std::cout << "Result:\n\tIter: " << iter << "\n\tError: " << error << "\n\tTime: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Чистка памяти
	cudaFree(A_Device);
	cudaFree(A_new_Device);
	cudaFree(A_error_Device);
	cudaFree(tempStorage);

	MPI_Finalize();

	return 0;
}