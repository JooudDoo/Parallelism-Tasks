#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>

#define arraySize 10000000

#ifndef nThreads
#define nThreads 80
#endif

#ifdef Double
using floatingType = double;
#else
using floatingType = float;
#endif

int main(int argc, char const *argv[])
{
    double pi = acos(-1);
    floatingType sum = 0;
    floatingType* array = new floatingType[arraySize];

    auto allProgramStart = std::chrono::high_resolution_clock::now();


#ifdef CPU
    auto allCycles = std::chrono::high_resolution_clock::now();
    auto firstCycle = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for num_threads(nThreads)
    for(int i = 0; i < arraySize; i++){
        array[i] = sin(2 * pi*i/arraySize);
    }

#ifdef CPU
    auto firstCycleElapsed = std::chrono::high_resolution_clock::now() - firstCycle;
    auto secondCycle = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for reduction(+:sum) num_threads(nThreads)
    for(int i = 0; i < arraySize; i++){
        sum += array[i];
    }

#ifdef CPU
    auto allCyclesElapsed = std::chrono::high_resolution_clock::now() - allCycles;
    auto secondCycleElapsed = std::chrono::high_resolution_clock::now() - secondCycle;
#endif

    auto allProgramElapsed = std::chrono::high_resolution_clock::now() - allProgramStart;
    long long programMicro = std::chrono::duration_cast<std::chrono::microseconds>(allProgramElapsed).count();
    std::cout << std::fixed;
#ifdef CPU
    long long allCyclesMicro = std::chrono::duration_cast<std::chrono::microseconds>(allCyclesElapsed).count();
    long long firstCyclesMicro = std::chrono::duration_cast<std::chrono::microseconds>(firstCycleElapsed).count();
    long long secondCyclesMicro = std::chrono::duration_cast<std::chrono::microseconds>(secondCycleElapsed).count();
    std::cout << "Two cycles " << allCyclesMicro << " us" << std::endl;
    std::cout << "First cycle " << firstCyclesMicro << " us" << std::endl;
    std::cout << "Second cycle " << secondCyclesMicro << " us" << std::endl;
#endif
    std::cout << "Programm " << programMicro << " us" << std::endl;
    std::cout << std::setprecision(20) << sum << std::endl;

    delete[] array;

    return 0;
}