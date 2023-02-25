#include <iostream>
#include <cstring>



#define at(arr, x, y) (arr[(x)*n+(y)]) 

int main(){
    constexpr int n = 1024;
    constexpr int m = n;
    constexpr double eps = 1E-6;
    constexpr int iterations = 1E6;

//Создаем матрицу
   double* F = new double[n*m];
   double* Fnew = new double[n*m];

    std::memset(F, 0, sizeof(double)*(n)*(m));
    at(F, 0, 0) = 10;
    at(F, 0, m-1) = 20;
    at(F, n-1, 0) = 20;
    at(F, n-1, m-1) = 30;
    for(int i = 1; i < n-1; i++){
        at(F,0,i) = (at(F,0,m-1)-at(F,0,0))/(m-1)*i+at(F,0,0);
        at(F,i,0) = (at(F,n-1,0)-at(F,0,0))/(n-1)*i+at(F,0,0);

        at(F,n-1,i) = (at(F,n-1,m-1)-at(F,n-1,0))/(m-1)*i+at(F,n-1,0);
        at(F,i,m-1) = (at(F,n-1,m-1)-at(F,0,m-1))/(m-1)*i+at(F,0,m-1);
    }
    std::memcpy(Fnew, F, sizeof(double)*(n)*(m));

    double error = 1;
    int iteration = 0;
    while(iteration < iterations && error > eps){
        error = 0;

        for(int x = 1; x < n-1; x++){
            for(int y = 1; y < m-1; y++){
                at(Fnew,x,y) = 0.25 * (at(F, x+1,y) + at(F,x-1,y) + at(F,x,y-1) + at(F,x,y+1));
                error = std::max(error, std::abs(at(Fnew,x,y) - at(F,x,y)));
            }
        }

        std::memcpy(F, Fnew, sizeof(double)*(n)*(m));
        if(iteration % 100 == 0){
            std::cout << iteration << '-' << "error: " << error << std::endl;
        }
        
        iteration += 1;
    }


    for(int x = 0; x < n; x++){
        for(int y = 0; y < m; y++){
            std::cout << at(F,x, y) << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}