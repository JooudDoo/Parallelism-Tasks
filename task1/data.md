# GPU

## float

Programm 482824 us

> result:  -0.028534

   Time(%)  | Time |    Calls  |   Avg  |       Min  |       Max  | Name 
-----------|------|-----------|--------|------------|------------|------
38.06% |  131.78us |        1 | 131.78us|  131.78us|  131.78us|  main_24_gpu  
34.63% |  119.90us |        1 | 119.90us|  119.90us|  119.90us|  main_36_gpu  
25.80% |  89.343us |        1 | 89.343us|  89.343us|  89.343us|  main_36_gpu__red  
0.92%  | 3.2000us  |       1  |3.2000us | 3.2000us | 3.2000us | [CUDA memcpy DtoH]  
0.59%  | 2.0480us  |       1  |2.0480us | 2.0480us | 2.0480us | [CUDA memcpy HtoD]  

## double

Programm 479460 us 

> result: -0.000000

   Time(%)  | Time |    Calls  |   Avg  |       Min  |       Max  | Name 
-----------|------|-----------|--------|------------|------------|------
35.85% |  131.71us |      1  | 131.71us  | 131.71us  | 131.71us |  main_36_gpu
35.70% |  131.17us |      1  | 131.17us  | 131.17us  | 131.17us |  main_24_gpu
27.17% |  99.839us |      1  | 99.839us  | 99.839us  | 99.839us |  main_36_gpu__red  
0.78%  | 2.8480us  |      1   |2.8480us   |2.8480us   |2.8480us  | [CUDA memcpy DtoH]  
0.51%  | 1.8560us  |      1   |1.8560us   |1.8560us   |1.8560us  | [CUDA memcpy HtoD]  


# CPU

## One core

## float

Two cycles 47310 us  
First cycle 43695 us  
Second cycle 3615 us  
Programm 47311 us  
-0.043014  

> result: -0.043014

## double

Two cycles 67642 us  
First cycle 60046 us  
Second cycle 7595 us  
Programm 67642 us  
-0.000000  

> 0.000000


## Multi core

## float

Two cycles 22553 us  
First cycle 17858 us  
Second cycle 4695 us  
Programm 22554 us  
-0.043014  

> result: -0.043014


## double

Two cycles 28765 us  
First cycle 17811 us  
Second cycle 10953 us  
Programm 28766 us  
-0.000000  

> -0.000000
