#include "MST.cuh"

#include <stdexcept>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(){
    throw std::runtime_error("You should specifed in_size and out_size");
}

FullyConnectedLayer::FullyConnectedLayer(const int in_size, const int out_size, const bool use_bias, const int seed) : in_size(in_size), out_size(out_size), use_bias(use_bias) {
    ReLU_weight_init__(weights_device, in_size, out_size, seed);
    if(use_bias){
        ReLU_weight_init__(bias_deivice, 1, out_size, seed);
    }
}

FullyConnectedLayer::~FullyConnectedLayer(){
    if(weights_device){
        cudaFree(weights_device);
    }
    if(bias_deivice){
        cudaFree(bias_deivice);
    }
}

FullyConnectedLayer::forward(const floatingType* x){
    
}

/*
* Инициализирует веса с помощью метода Kaiming he
* Принимает указатель на массив на видеокарте для копирования весов на него
*/
void FullyConnectedLayer::ReLU_weight_init__(floatingType* arr_d, const int in_size, const int out_size, const int random_seed){
    std::default_random_engine rangen;
    if(random_seed != -1){
        rangen.seed(random_seed);
    }

    floatingType disp = std::sqrt((double)2/out_size);
    floatingType mean = 0;

    std::normal_distribution<floatingType> distribution(mean, disp);

    if(arr_d == nullptr){
        cudaMalloc(&arr_d, in_size*out_size*sizeof(floatingType));
    }

    floatingType* arr = (floatingType*)malloc(in_size*out_size*sizeof(floatingType));

    for(size_t i = 0; i < in_size*out_size; i++){
        arr[i] = distribution(rangen);
    }

    cudaMemcpy(arr_d, arr, in_size*out_size*sizeof(floatingType), cudaMemcpyHostToDevice);

    free(arr);
}