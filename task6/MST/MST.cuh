#pragma once

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include "MST_Layers.cuh"

#include <fstream>

template <typename T, typename... Args>
std::unique_ptr<T> create_layer(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

static void load_binary_file(calc_type* target, const std::string file_path, const std::pair<int, int> size){
    std::ifstream file(file_path, std::ios::binary);

    calc_type* file_info;
    cudaMallocHost(&file_info, size.first*size.second*sizeof(calc_type));
    std::ifstream file_input(file_path, std::ios::binary);
    for(size_t i = 0; i < size.first*size.second; file >> file_info[i++]);

    calc_type* arr_host;
    cudaMallocHost(&arr_host, size.first*size.second*sizeof(calc_type));
    for (int i = 0; i < size.first; ++i) {
        for (int j = 0; j < size.second; ++j) {
            arr_host[i * size.second + j] = file_info[i * size.second + j];
        }
    }

    cudaMemcpy(target, arr_host, size.first*size.second*sizeof(calc_type), cudaMemcpyHostToDevice);
    
    cudaFreeHost(file_info);
    cudaFreeHost(arr_host);
}