#include <iostream>
#include <sstream>
#include <cstring>

/*
Contains functions to process comand line arguments and show it in default output stream
*/

typedef struct commandArguments_S{
    bool showResultArr;
    bool initUsingMean;
    double eps;
    int iterations;
    int n;
    int m;
} cmdArgs;

template<typename T>
T extractArgument(char* arr){
    std::stringstream stream;
    stream << arr;
    T result;
    if (!(stream >> result)){
        throw std::invalid_argument("Wrong argument type");
    }
    return result;
}

void processArgs(int argc, char *argv[], cmdArgs* args){
    for(int arg = 0; arg < argc; arg++){
        if(std::strcmp(argv[arg], "-eps") == 0){
            args->eps = extractArgument<double>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-i") == 0){
            args->iterations = extractArgument<int>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-s") == 0){
            args->n = extractArgument<int>(argv[arg+1]);
            args->m = args->n;
            arg++;
        }
        else if(std::strcmp(argv[arg], "-show") == 0){
            args->showResultArr = true;
        }
        else if(std::strcmp(argv[arg], "-O") == 0){
            args->initUsingMean = true;
        }
    }
}

constexpr const char* boolToString(const bool& val){
    if(val){
        return "Yes";
    }
    return "No";
}

void printSettings(cmdArgs* args){
    std::cout << "Current settings:" << std::endl;
    std::cout << "\tEPS: " << args->eps << std::endl;
    std::cout << "\tMax iteration: " << (double)(args->iterations) << std::endl;
    std::cout << "\tSize: " << args->n << 'x' << args->m << std::endl;
    std::cout << "\tInit grid with mean of angles: " << boolToString(args->initUsingMean) << std::endl;
    std::cout << "\tShow output array at end of calculation: " << boolToString(args->showResultArr) << std::endl << std::endl;
}