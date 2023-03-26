#include <iostream>
#include <cstring>
#include <sstream>
#include <math.h>
#include <cmath>

#ifdef OPENACC__
#include <openacc.h>
#endif
#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#include <cublas_v2.h>

#define at(arr, x, y) (arr[(x)*n+(y)]) 

constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

void initArrays(double* mainArr, double* subArr, int& n, int& m, bool initUsingMean){
    std::memset(mainArr, 0, sizeof(double)*(n)*(m));

    for(int i = 0; i < n*n && initUsingMean; i++){
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
    std::memcpy(subArr, mainArr, sizeof(double)*(n)*(m));
}

template<typename T>
T extractNumber(char* arr){
    std::stringstream stream;
    stream << arr;
    T result;
    if (!(stream >> result)){
        throw std::invalid_argument("Wrong argument type");
    }
    return result;
}

constexpr int ITERS_BETWEEN_UPDATE = 250;
constexpr double negOne = -1;

int main(int argc, char *argv[]){

    bool showResultArr = false;
    bool initUsingMean = false;
    double eps = 1E-6;
    int iterations = 1E6;
    int n = 10;
    int m = n;

    for(int arg = 0; arg < argc; arg++){
        if(std::strcmp(argv[arg], "-eps") == 0){
            eps = extractNumber<double>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-i") == 0){
            iterations = extractNumber<int>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-s") == 0){
            n = extractNumber<int>(argv[arg+1]);
            m = n;
            arg++;
        }
        else if(std::strcmp(argv[arg], "-show") == 0){
            showResultArr = true;
        }
        else if(std::strcmp(argv[arg], "-O") == 0){
            initUsingMean = true;
        }
    }

    std::cout << "Current settings:" << std::endl;
    std::cout << "\tEPS: " << eps << std::endl;
    std::cout << "\tMax iteration: " << iterations << std::endl;
    std::cout << "\tSize: " << n << 'x' << m << std::endl << std::endl;
 
    double* F = new double[n*m];
    double* Fnew = new double[n*m];

    double* inter = new double[n*m];

    initArrays(F, Fnew, n, m, initUsingMean);

    double error = 1;
    int iteration = 0;

    int itersBetweenUpdate = 0;

    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasStatus_t status;
    int max_idx = 0;

    #pragma acc enter data copyin(Fnew[:n*m], F[:n*m], inter[:n*m])

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif
    do {
        #pragma acc parallel loop collapse(2) present(Fnew[:n*m], F[:n*m]) vector_length(128) async
        for(int x = 1; x < n-1; x++){
            for(int y= 1; y < m-1; y++){
                at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
            }
        }

        double* swap = F;
        F = Fnew;
        Fnew = swap;

#ifdef OPENACC__
        acc_attach((void**)F);
        acc_attach((void**)Fnew);
#endif

        if(itersBetweenUpdate >= ITERS_BETWEEN_UPDATE && iteration < iterations){
            
            #pragma acc data present(inter[:n*m], Fnew[:n*m], F[:n*m]) wait
            {
                #pragma acc host_data use_device(Fnew, F, inter)
                {

                    status = cublasDcopy(handle, n*m, F, 1, inter, 1);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "copy error" << std::endl, exit(30);

                    status = cublasDaxpy(handle, n*m, &negOne, Fnew, 1, inter, 1);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "sum error" << std::endl, exit(40);
                    
                    status = cublasIdamax(handle, n*m, inter, 1, &max_idx);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "abs max error" << std::endl, exit(41);
                }
            }

            #pragma acc update self(inter[max_idx-1])
            error = fabs(inter[max_idx-1]);

            itersBetweenUpdate = -1;
        }
        iteration++;
        itersBetweenUpdate++;
    } while(iteration < iterations && error > eps);

#ifdef NVPROF_
    nvtxRangePop();
#endif

    #pragma acc exit data delete(Fnew[:n*m]) copyout(F[:n*m])
    cublasDestroy(handle);

    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Error: " << error << std::endl;
    for(int x = 0; x < n && showResultArr; x++){
        for(int y = 0; y < m; y++){ 
            std::cout << at(F,x, y) << ' ';
        }
        std::cout << std::endl;
    }

    delete[] F;
    delete[] Fnew;

    return 0;
}