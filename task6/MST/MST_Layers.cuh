#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <vector>


#include <cublas_v2.h>
#include "cuda_runtime.h"

#include <memory>
#include <unordered_map>

enum class PARAMETR_SOURCE{
    FILE, GENERATE
};

enum class ARRAY_SOURCE{
    DEVICE, HOST
};

using calc_type = float;

#include "MDT_array.cuh"

class Layer {
public:
    virtual MDT_array<calc_type>& forward(MDT_array<calc_type>& input) = 0;
    MDT_array<calc_type>& operator()(MDT_array<calc_type>& input){
        return forward(input);
    }
};

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(int input_size, int output_size, std::string weigth_path = "", std::string bias_path = "");

    void set_weights(const PARAMETR_SOURCE source_type, const std::string file_path = "");
    void set_bias(const PARAMETR_SOURCE source_type, const std::string file_path = "");

    ~FullyConnectedLayer();

    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;

    int get_size();
    std::pair<int, int> get_dims();

private:
    void generate_parmeter(calc_type* target, const std::pair<int, int> size, const int random_seed = -1);
    void load_parameter(calc_type* target, const std::string file_path, const std::pair<int, int> size);

    int input_size_;
    int output_size_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    MDT_array<calc_type>* d_output_;
    calc_type* d_weight_;
    calc_type* d_bias_;
    cublasHandle_t handle_;
};

using FC = FullyConnectedLayer;

class Sigmoid : public Layer {
    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;
};

MDT_array<calc_type>& Sigmoid_F(MDT_array<calc_type>& arr);

class Sequential : public Layer {
public:

    Sequential();

    void add_layer(std::unique_ptr<Layer> layer);

    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;

private:
    std::vector<std::unique_ptr<Layer>> layers;
};