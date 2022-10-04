#include "stdio.h"
#include "stdlib.h"
#include "omp.h"
#include "time.h"

void addition_parallel(int n) {

    double arr1[n], arr2[n], arr3[n];
    srand(time(0));

    int num_threads;
    double startTime = omp_get_wtime();
    
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
        #pragma omp for
        for (int i=0; i<n; i++)
        {
            arr1[i] = rand()*12.472910292012446683;
            arr2[i] = rand()*12.472910292012446683;
            arr3[i] = arr1[i] + arr2[i];
        }   
    }

    double endTime = omp_get_wtime();
    // printf(" Parallel = %f\n", endTime-startTime);
    printf("%d,%f\n",num_threads,endTime-startTime);

}


void addition_serial(int n){

    double arr1[n], arr2[n], arr3[n];
    srand(time(0));

    double startTime = omp_get_wtime();
    
    for (int i=0; i<n; i++) {
        arr1[i] = rand()*12.472910292012446683;
        arr2[i] = rand()*12.472910292012446683;
        arr3[i] = arr1[i] + arr2[i];
    }   

    double endTime = omp_get_wtime();
    // printf("\n Serial = %f\n", endTime-startTime);

}


int main() {
    int n=10000;
    addition_serial(n);
    addition_parallel(n);
    return 0;
}
