#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "Lab3IO.h"
#include "timer.h"

#define TOL 0.0005



int main(int argc, char* argv[]) {

	int i, j, k, size;
	double** Au;
	double* X;
	double temp, error, Xnorm;
    double start; double end;
	int* index;
	FILE* fp;

	int threadCount = atoi(argv[1]);

	/*Load the datasize and verify it*/
	Lab3LoadInput(&Au, &size);
	if ((fp = fopen("data_input","r")) == NULL){
		printf("Fail to open the result data file!\n");
		return 2;
	}
	fscanf(fp, "%d\n\n", &i);
	if (i != size){
		printf("The problem size of the input file and result file does not match!\n");
		return -1;
	}

	/*Calculate the solution by serial code*/
	X = CreateVec(size);
    index = malloc(size * sizeof(int));
    printf("size: %d\n", size);
    GET_TIME(start);

    #pragma omp parallel for
    for (i = 0; i < size; ++i)
        index[i] = i;

    
    if (size == 1)
        X[0] = Au[0][1] / Au[0][0];
    else{
        /*Gaussian elimination*/
        for (k = 0; k < size - 1; ++k){
            /*Pivoting*/
            temp = 0;
            for (i = k, j = 0; i < size; ++i)
                if (temp < Au[index[i]][k] * Au[index[i]][k]){
                    temp = Au[index[i]][k] * Au[index[i]][k];
                    j = i;
                }
            if (j != k)/*swap*/{
                i = index[j];
                index[j] = index[k];
                index[k] = i;
            }
            /*calculating*/ 
            #pragma omp parallel for schedule(dynamic) num_threads(threadCount) shared(Au, index, k, size) private(i, j, temp)
            for (i = k + 1; i < size; ++i){
                temp = Au[index[i]][k] / Au[index[k]][k];
                for (j = k; j < size + 1; ++j)
                    Au[index[i]][j] -= Au[index[k]][j] * temp;
            }       
        }
        /*Jordan elimination*/
        
        for (k = size - 1; k > 0; --k){
            #pragma omp parallel for schedule(dynamic) num_threads(threadCount) shared(Au, index, k, size) private(i, temp)
            for (i = k - 1; i >= 0; --i ){
                temp = Au[index[i]][k] / Au[index[k]][k];
                Au[index[i]][k] -= temp * Au[index[k]][k];
                Au[index[i]][size] -= temp * Au[index[k]][size];
            } 
        }
        /*solution*/
        #pragma omp parallel for
        for (k=0; k< size; ++k)
            X[k] = Au[index[k]][size] / Au[index[k]][k];
    }
    
    GET_TIME(end);
    Lab3SaveOutput(X, size, end-start);

}