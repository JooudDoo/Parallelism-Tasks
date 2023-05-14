#pragma once

template<typename data_type>
class MDT_array{
public:

    MDT_array();
    MDT_array(int, data_type*, ARRAY_SOURCE src_type = ARRAY_SOURCE::HOST);
    MDT_array(int);
    ~MDT_array();

    __device__ data_type& operator[] (int index); // Получение I-ого элемента на видеокарте
    data_type operator() (int index); // Получение I-ого элемента и копирование его сразу на хост
    data_type* operator&() const; // Отдает указатель на массив на видеокарта
    int get_size();
private:
    data_type* array_;
    int size_;
};

template<typename data_type>
MDT_array<data_type>::MDT_array(int size, data_type* src, ARRAY_SOURCE src_type) : size_(size){
    cudaMalloc(&array_, size_*sizeof(data_type));
    if(src_type == ARRAY_SOURCE::DEVICE){
        cudaMemcpy(array_, src, sizeof(data_type) * size, cudaMemcpyDeviceToDevice);
    }
    else if(src_type == ARRAY_SOURCE::HOST){
        cudaMemcpy(array_, src, sizeof(data_type) * size, cudaMemcpyHostToDevice);
    }
}

template<typename data_type>
MDT_array<data_type>::MDT_array(){
     throw std::runtime_error("You should specifed size");
}

template<typename data_type>
MDT_array<data_type>::MDT_array(int size) : size_(size){
    cudaMalloc(&array_, size_*sizeof(data_type));
}

template<typename data_type>
MDT_array<data_type>::~MDT_array(){
    cudaFree(array_);
}

template<typename data_type>
__device__ data_type& MDT_array<data_type>::operator[](int index){
    return array_[index];
}

template<typename data_type>
data_type* MDT_array<data_type>::operator&() const{
    return array_;
}

template<typename data_type>
data_type MDT_array<data_type>::operator()(int index){
    data_type tmp;
    cudaMemcpy(&tmp, array_ + index, sizeof(data_type), cudaMemcpyDeviceToHost);
    return tmp;
}

template<typename data_type>
int MDT_array<data_type>::get_size(){
    return size_;
}