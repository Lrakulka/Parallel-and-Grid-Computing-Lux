/*
* Author Oleksandr Borysov
* Task3
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <ctime>

#define MAX 32767
#define PLOT_DATA_FILE "plot_data3.txt"
#define SEED 254321

__global__ void getCounts(double* results, unsigned long* idxSteps, unsigned long* steps) {
	double x, y, z;
	unsigned long count = 0;
	// index of thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(SEED + idx, 0, 0, &state);

	for (unsigned long i = 0; i < idxSteps[idx]; ++i) {
		x = ((double)((curand(&state)) % MAX)) / MAX;
		y = ((double)((curand(&state)) % MAX)) / MAX;
		z = sqrt((x * x) + (y * y));
		if (z <= 1) {
			++count;
		}
	}
	results[idx] = ((double)count / *steps) * 4.0;
}

int main(int argc, char* argv[]) {
	unsigned long stepNumber, threadBlock, threads, threadSteps, threadNumber;
	unsigned long *steps;
	double *results;
	cudaError_t cudaStatus;
	clock_t begin = 0;
	
	printf("Type number of steps \n");
	scanf("%lu", &stepNumber);  
	printf("Thread blocks \n");
	scanf("%lu", &threadBlock);  
	printf("Threads in block \n");
	scanf("%lu", &threads); 

	// stepNumber = 1000000; threadBlock = 10; threads = 10;

	begin = clock();
	threadNumber = threadBlock * threads;
	threadSteps = stepNumber / threadNumber;
	results = (double*) calloc(threadNumber, sizeof(double));
	steps = (unsigned long*) calloc(threadNumber, sizeof(long));
	for (int i = 0; i < threadNumber - 1; ++i) {
		steps[i] = threadSteps;
	}
	steps[threadNumber - 1] = stepNumber - threadSteps * (threadNumber - 1);

	unsigned long *d_steps, *d_stepsNumber;
	double *d_results;

	//----------------
	cudaStatus = cudaMalloc(&d_results, sizeof(double) * threadNumber);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_results failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc(&d_steps, sizeof(long) * threadNumber);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_steps failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc(&d_stepsNumber, sizeof(long));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_stepsNumber failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(d_steps, steps, sizeof(long) * threadNumber, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_steps failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(d_stepsNumber, &stepNumber, sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_stepsNumber failed!");
        goto Error;
    }
	// run CUDA method
	getCounts <<<threadBlock, threads>>>(d_results, d_steps, d_stepsNumber);

	cudaStatus = cudaMemcpy(results, d_results, sizeof(double) * threadNumber, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_result failed! code %d", cudaStatus);
        goto Error;
    }


Error:
	cudaFree(d_results);
	cudaFree(d_steps);
	cudaFree(d_stepsNumber);

	if (cudaStatus == 0) {
		double time_spent = (double) (clock() - begin) / CLOCKS_PER_SEC;
		double pi = 0;
		for (unsigned long i = 0; i < threadNumber; ++i) {
			pi += results[i];
		}
		printf("Calculated PI is = %f.\n Time= %f\n", pi, time_spent);
		FILE* dataPlotFile;
		dataPlotFile = fopen(PLOT_DATA_FILE, "a");
		fprintf(dataPlotFile, "%d %f %d\n", threadNumber, time_spent, stepNumber);
		fclose(dataPlotFile);
	}
	free(results);
	free(steps);
	return cudaStatus;
}