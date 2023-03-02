#include <iostream>
#include <cstring>
#include <sstream>
#include <math.h>
#include <cmath>

#include <openacc.h>

#define at(arr, x, y) (arr[(x)*n+(y)]) 

void initArrays(double* mainArr, double* subArr, int& n, int& m){
    std::memset(mainArr, 0, sizeof(double)*(n)*(m));
    at(mainArr, 0, 0) = 10;
    at(mainArr, 0, m-1) = 20;
    at(mainArr, n-1, 0) = 20;
    at(mainArr, n-1, m-1) = 30;
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

int main(int argc, char *argv[]){

    bool showResultArr = false;
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
    }

    std::cout << "Current settings:" << std::endl;
    std::cout << "\tEPS: " << eps << std::endl;
    std::cout << "\tMax iteration: " << iterations << std::endl;
    std::cout << "\tSize: " << n << 'x' << m << std::endl << std::endl;
 
    double* F = new double[n*m];
    double* Fnew = new double[n*m];

    initArrays(F, Fnew, n, m);

    double error = 0;
    int iteration = 0;

    #pragma acc enter data copyin(Fnew[:n*m], F[:n*m])

    do {
        error = 0;

        #pragma acc parallel loop collapse(2) present(Fnew[:n*m], F[:n*m]) reduction(max:error) vector_length(128)
        for(int x = 1; x < n-1; x++){
            for(int y= 1; y < m-1; y++){
                at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
                error = fmax(error, fabs(at(Fnew,x,y) - at(F,x,y)));
            }
        }

        double* swap = F;
        F = Fnew;
        Fnew = swap;

        acc_attach((void**)F);
        acc_attach((void**)Fnew);

        iteration++;
    } while(iteration < iterations && error > eps);

{

    // #pragma acc parallel
    // do {
    //     error = 0;

    //     #pragma acc loop collapse(2) reduction(max:error) independent
    //     for(int x = 1; x < n-1; x++){
    //         for(int y= 1; y < m-1; y++){
    //             at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
    //             error = fmax(error, fabs(at(Fnew,x,y) - at(F,x,y)));
    //         }
    //     }

    //     double* swap = F;
    //     F = Fnew;
    //     Fnew = swap;

    //     #pragma acc atomic
    //     iteration++;
    // } while(iteration < iterations && error > eps);


    //    do {
    //     error = 0;

    //     #pragma acc parallel loop collapse(2) present(Fnew[:n*m], F[:n*m]) reduction(max:error)
    //     for(int x = 1; x < n-1; x++){
    //         for(int y= 1; y < m-1; y++){
    //             at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
    //             error = fmax(error, fabs(at(Fnew,x,y) - at(F,x,y)));
    //         }
    //     }

    //     double* swap = F;
    //     F = Fnew;
    //     Fnew = swap;

    //     acc_attach((void**)F);
    //     acc_attach((void**)Fnew);

    //     iteration++;
    // } while(iteration < iterations && error > eps);

    // #pragma acc parallel copy(error) present(F[:m*n], Fnew[:m*n]), reduction(max:error)
    // #pragma acc loop
    // do {
    //     error = 0;

    //     #pragma acc parallel loop copy(error) present(F[:m*n], Fnew[:m*n]) reduction(max:error)
    //     for(int i = n+1; i < n*(m-1); i += 1 + ((i+2)%n==0)*2){
    //         Fnew[i] = 0.25 * (F[i-1] + F[i+1] + F[i-m] + F[i+m]);
    //         error = max(error, abs(Fnew[i] - F[i]));
    //     }

    //     // std::memcpy(F, Fnew, sizeof(double)*(n)*(m));
    //     acc_memcpy_device(acc_deviceptr(F), acc_deviceptr(Fnew), sizeof(double)*n*m);

    //     // if(iteration % 1000 == 0){
    //     //     std::cout << "[" << iteration << "] error - " << error << std::endl; 
    //     // }
    //     #pragma acc atomic update
    //     iteration++;
    // } while((iteration < iterations && error > eps));

    // #pragma acc parallel present(F[:m*n], Fnew[:m*n], iteration) 
    // for(iteration = 0; iteration < iterations; iteration++){
    //     #pragma acc loop independent
    //     for(int i = n+1; i < n*(m-1); i += 1 + ((i+2)%n==0)*2){
    //         Fnew[i] = 0.25 * (F[i-1] + F[i+1] + F[i-m] + F[i+m]);
    //         error = max(error, abs(Fnew[i] - F[i]));
    //     }
    //     #pragma acc loop independent wait
    //     for(int i = 0; i < n*m; i++){
    //         F[i] = Fnew[i];
    //     }
    // }
}
  
    #pragma acc exit data delete(Fnew[:n*m]) copyout(F[:n*m])

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