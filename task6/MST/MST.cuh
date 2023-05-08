#pragma once

#include <iostream>

template<typename arr_type, int row_size_temp>
inline arr_type at_d(arr_type* arr, int y, int x, int row_size = row_size_temp){
    return arr[y*row_size + x];
}

template<typename arr_type>
inline arr_type at_d(arr_type* arr, int y, int x, int row_size){
    return arr[y*row_size + x];
}

using floatingType = double;

class FullyConnectedLayer {

public:
    FullyConnectedLayer();
    FullyConnectedLayer(const int in_size_, const int out_size_, const bool use_bias = true, const int seed = 42);
    ~FullyConnectedLayer();

    floatingType* forward(const floatingType* x);

private:
    int in_size;
    int out_size;
    bool use_bias;

    floatingType* weights_device = nullptr;
    floatingType* bias_deivice = nullptr;
    void ReLU_weight_init__(floatingType* arr_d, const int in_size, const int out_size, const int random_seed = -1);
};