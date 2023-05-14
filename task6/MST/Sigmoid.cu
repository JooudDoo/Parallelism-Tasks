#include "MST.cuh"

__global__ void sigmoid_cuda(calc_type* arr, int n){
    if(threadIdx.x < n){
        arr[threadIdx.x] = 1 / (1 + exp(-arr[threadIdx.x]));
    }
}

MDT_array<calc_type>& Sigmoid::forward(MDT_array<calc_type>& arr) {
    sigmoid_cuda<<<1, arr.get_size()>>>(&arr, arr.get_size());
    return arr;
}

MDT_array<calc_type>& Sigmoid_F(MDT_array<calc_type>& arr){
    return Sigmoid()(arr);
}