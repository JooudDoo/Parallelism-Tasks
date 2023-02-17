# GPU

## float

> result:  -0.028534

   Time(%)  | Time |    Calls  |   Avg  |       Min  |       Max  | Name 
-----------|------|-----------|--------|------------|------------|------
38.06% |  131.78us |        1 | 131.78us|  131.78us|  131.78us|  main_20_gpu  
34.63% |  119.90us |        1 | 119.90us|  119.90us|  119.90us|  main_27_gpu  
25.80% |  89.343us |        1 | 89.343us|  89.343us|  89.343us|  main_27_gpu__red  
0.92%  | 3.2000us  |       1  |3.2000us | 3.2000us | 3.2000us | [CUDA memcpy DtoH]  
0.59%  | 2.0480us  |       1  |2.0480us | 2.0480us | 2.0480us | [CUDA memcpy HtoD]  

## double

> result: -0.000000

   Time(%)  | Time |    Calls  |   Avg  |       Min  |       Max  | Name 
-----------|------|-----------|--------|------------|------------|------
35.85% |  131.71us |        1  | 131.71us  | 131.71us  | 131.71us |  main_27_gpu
35.70% |  131.17us |        1  | 131.17us  | 131.17us  | 131.17us |  main_20_gpu
27.17% |  99.839us |        1  | 99.839us  | 99.839us  | 99.839us |  main_27_gpu__red  
0.78%  | 2.8480us  |       1   |2.8480us   |2.8480us   |2.8480us  | [CUDA memcpy DtoH]  
0.51%  | 1.8560us  |       1   |1.8560us   |1.8560us   |1.8560us  | [CUDA memcpy HtoD]  


# CPU

## float

> result: -0.027786

> result: 178548 us

## double

> 0.000000

> 203043 us