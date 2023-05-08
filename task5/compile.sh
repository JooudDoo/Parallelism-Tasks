mpic++ t5_MPI.cu -o t_MPI -O2 -D DEBUG -D THREAD_ANALOG_MODE
mpic++ t5_nccl.cu -o t_nccl -O2 -D DEBUG -D THREAD_ANALOG_MODE -lnccl