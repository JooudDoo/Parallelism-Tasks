#pragma once

#include <iostream>
#include <string>
#include <tuple>

#include <cublas_v2.h>
#include "cuda_runtime.h"

enum class PARAMETR_SOURCE{
    FILE, GENERATE
};

using calc_type = float;

class FullyConnectedLayer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(int input_size, int output_size);

    void set_weights(const PARAMETR_SOURCE source_type, const std::string file_path = "");
    void set_bias(const PARAMETR_SOURCE source_type, const std::string file_path = "");

    ~FullyConnectedLayer();

    calc_type* forward(const calc_type* input);

private:
    void generate_parmeter(calc_type* target, const std::pair<int, int> size, const int random_seed = -1);
    void load_parameter(calc_type* target, const std::string file_path, const std::pair<int, int> size);

    int input_size_;
    int output_size_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    calc_type* d_input_;
    calc_type* d_output_;
    calc_type* d_weight_;
    calc_type* d_bias_;
    cublasHandle_t handle_;
};