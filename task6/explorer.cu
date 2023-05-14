#include "MST/MST.cuh"

#include <iostream>

#ifdef NVPROF_
#include <nvToolsExt.h>
#endif

int main(){
    
    Sequential Net;

    MDT_array<calc_type> a {1024};

    load_binary_file(&a, "torch/input", {1, 1024});
    
    Net.add_layer(create_layer<FC>(1024, 256, "torch/fc1_weights", "torch/fc1_bias"));
    Net.add_layer(create_layer<Sigmoid>());


    Net.add_layer(create_layer<FC>(256, 16, "torch/fc2_weights", "torch/fc2_bias"));
    Net.add_layer(create_layer<Sigmoid>());

    Net.add_layer(create_layer<FC>(16, 1, "torch/fc3_weights", "torch/fc3_bias"));
    Net.add_layer(create_layer<Sigmoid>());


#ifdef NVPROF_
    nvtxRangePush("Forward_pass");
#endif
    MDT_array<calc_type> b = Net.forward(a);
#ifdef NVPROF_
    nvtxRangePop();
#endif

    std::cout << b(0) << std::endl;

    return 0;
}