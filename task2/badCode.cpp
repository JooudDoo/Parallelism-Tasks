{

    // #pragma acc parallel
    // do {
    //     error = 0;

    //     #pragma acc loop collapse(2) reduction(max:error) independent
    //     for(int x = 1; x < n-1; x++){
    //         for(int y= 1; y < m-1; y++){
    //             at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
    //             error = fmax(error, fabs(at(Fnew,x,y) - at(F,x,y)));
    //         }
    //     }

    //     double* swap = F;
    //     F = Fnew;
    //     Fnew = swap;

    //     #pragma acc atomic
    //     iteration++;
    // } while(iteration < iterations && error > eps);


    //    do {
    //     error = 0;

    //     #pragma acc parallel loop collapse(2) present(Fnew[:n*m], F[:n*m]) reduction(max:error)
    //     for(int x = 1; x < n-1; x++){
    //         for(int y= 1; y < m-1; y++){
    //             at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
    //             error = fmax(error, fabs(at(Fnew,x,y) - at(F,x,y)));
    //         }
    //     }

    //     double* swap = F;
    //     F = Fnew;
    //     Fnew = swap;

    //     acc_attach((void**)F);
    //     acc_attach((void**)Fnew);

    //     iteration++;
    // } while(iteration < iterations && error > eps);

    // #pragma acc parallel copy(error) present(F[:m*n], Fnew[:m*n]), reduction(max:error)
    // #pragma acc loop
    // do {
    //     error = 0;

    //     #pragma acc parallel loop copy(error) present(F[:m*n], Fnew[:m*n]) reduction(max:error)
    //     for(int i = n+1; i < n*(m-1); i += 1 + ((i+2)%n==0)*2){
    //         Fnew[i] = 0.25 * (F[i-1] + F[i+1] + F[i-m] + F[i+m]);
    //         error = max(error, abs(Fnew[i] - F[i]));
    //     }

    //     // std::memcpy(F, Fnew, sizeof(double)*(n)*(m));
    //     acc_memcpy_device(acc_deviceptr(F), acc_deviceptr(Fnew), sizeof(double)*n*m);

    //     // if(iteration % 1000 == 0){
    //     //     std::cout << "[" << iteration << "] error - " << error << std::endl; 
    //     // }
    //     #pragma acc atomic update
    //     iteration++;
    // } while((iteration < iterations && error > eps));

    // #pragma acc parallel present(F[:m*n], Fnew[:m*n], iteration) 
    // for(iteration = 0; iteration < iterations; iteration++){
    //     #pragma acc loop independent
    //     for(int i = n+1; i < n*(m-1); i += 1 + ((i+2)%n==0)*2){
    //         Fnew[i] = 0.25 * (F[i-1] + F[i+1] + F[i-m] + F[i+m]);
    //         error = max(error, abs(Fnew[i] - F[i]));
    //     }
    //     #pragma acc loop independent wait
    //     for(int i = 0; i < n*m; i++){
    //         F[i] = Fnew[i];
    //     }
    // }
}
 