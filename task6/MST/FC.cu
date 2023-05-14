#include "MST.cuh"

#include <random>
#include <stdexcept>

FullyConnectedLayer::FullyConnectedLayer(){
    throw std::runtime_error("You should specifed in_size and out_size");
}

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, std::string weigth_path, std::string bias_path) : input_size_(input_size), output_size_(output_size){
    d_output_ = new MDT_array<calc_type>(output_size);
    cudaMalloc(&d_weight_, input_size_ * output_size_ * sizeof(calc_type));
    cudaMalloc(&d_bias_, output_size_ * sizeof(calc_type));
    cublasCreate(&handle_);

    if(weigth_path == ""){
        set_weights(PARAMETR_SOURCE::GENERATE);
    }
    else{
        set_weights(PARAMETR_SOURCE::FILE, weigth_path);
    }
    if(bias_path == ""){
        set_bias(PARAMETR_SOURCE::GENERATE);
    }
    else{
        set_bias(PARAMETR_SOURCE::FILE, bias_path);
    }

}

FullyConnectedLayer::~FullyConnectedLayer() {
    cublasDestroy(handle_);
    d_output_->~MDT_array();
    free(d_output_);
    cudaFree(d_weight_);
    cudaFree(d_bias_);
}

MDT_array<calc_type>& FullyConnectedLayer::forward(MDT_array<calc_type>& x) {
    calc_type* input = &x;
    // cublasSgemv(handle_, CUBLAS_OP_T, input_size_, output_size_, &alpha_, d_weight_, input_size_, input, 1, &beta_, &*d_output_, 1);
    // cublasSgeam(handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_size_, input_size_, &alpha_, d_bias_, output_size_, &alpha_, &*d_output_, output_size_, &*d_output_, output_size_);
    // cublasSaxpy(handle_, output_size_, &alpha_, d_bias_, 1, d_output_, 1);

    cublasSgemv(handle_, CUBLAS_OP_T, input_size_, output_size_, &alpha_, d_weight_, input_size_, input, 1, &beta_, &*d_output_, 1);
    cublasSaxpy(handle_, output_size_, &alpha_, d_bias_, 1, &*d_output_, 1);
    return *d_output_;
}

int FullyConnectedLayer::get_size(){
    return output_size_ * input_size_;
}
std::pair<int, int> FullyConnectedLayer::get_dims(){
    return {input_size_, output_size_};
}

void FullyConnectedLayer::set_weights(const PARAMETR_SOURCE source_type, const std::string file_path){
    if(source_type == PARAMETR_SOURCE::FILE){
        load_parameter(d_weight_, file_path, {input_size_, output_size_});
    }
    else if(source_type == PARAMETR_SOURCE::GENERATE){
        generate_parmeter(d_weight_, {input_size_, output_size_});
    }
}

void FullyConnectedLayer::set_bias(const PARAMETR_SOURCE source_type, const std::string file_path){
    if(source_type == PARAMETR_SOURCE::FILE){
        load_parameter(d_bias_, file_path, {1, output_size_});
    }
    else if(source_type == PARAMETR_SOURCE::GENERATE){
        generate_parmeter(d_bias_, {1, output_size_});
    }
}

void FullyConnectedLayer::generate_parmeter(calc_type* target, const std::pair<int, int> size, const int random_seed){
    int in_size = size.first;
    int out_size = size.second;
    
    std::default_random_engine rangen;
    if(random_seed != -1){
        rangen.seed(random_seed);
    }

    calc_type disp = std::sqrt((double)2/out_size);
    calc_type mean = 0;

    std::normal_distribution<calc_type> distribution(mean, disp);

    if(target == nullptr){
        cudaMalloc(&target, in_size*out_size*sizeof(calc_type));
    }

    calc_type* temp = (calc_type*)malloc(in_size*out_size*sizeof(calc_type));

    for(size_t i = 0; i < in_size*out_size; i++){
        temp[i] = distribution(rangen);
    }

    cudaMemcpy(target, temp, in_size*out_size*sizeof(calc_type), cudaMemcpyHostToDevice);

    free(temp);
}

void FullyConnectedLayer::load_parameter(calc_type* target, const std::string file_path, const std::pair<int, int> size){
    load_binary_file(target, file_path, size);
}