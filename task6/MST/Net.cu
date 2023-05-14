#include "MST.cuh"

Sequential::Sequential(){}

void Sequential::add_layer(std::unique_ptr<Layer> layer){
    layers.push_back(std::move(layer));
}

MDT_array<calc_type>& Sequential::forward(MDT_array<calc_type>& x){
    for(auto& lr : layers){
        x = lr->forward(x);
    }
    return x;
}