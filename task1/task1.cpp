// pgc++ task1.cpp -o t1.out -acc -Minfo=accel 
// nvprof ./t1.out

#include <iostream>
#include <cmath>
#include <chrono>

#define PGI_ACC_TIME 1

#define arraySize 10000000

#define DDouble

#ifdef DDouble
using floatingType = double;
#else
using floatingType = float;
#endif

int main(int argc, char const *argv[])
{
    double pi = acos(-1);
    floatingType sum = 0;
    floatingType* array = new floatingType[arraySize];

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc enter data create(array[0:arraySize], sum) copyin(pi)
    
    #pragma acc parallel loop present(array[0:arraySize],sum)
    for(int i = 0; i < arraySize; i++){
        array[i] = sin(2 * pi*i/arraySize);
    }

    #pragma acc parallel loop present(array[0:arraySize],sum) reduction(+:sum)
    for(int i = 0; i < arraySize; i++){
        sum += array[i];
    }

    #pragma acc exit data copyout(sum) delete(array[0:arraySize], pi)

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed).count();
    std::cout << std::fixed;
    std::cout << microseconds << std::endl;
    std::cout << sum << std::endl;

    delete[] array;

    return 0;
}