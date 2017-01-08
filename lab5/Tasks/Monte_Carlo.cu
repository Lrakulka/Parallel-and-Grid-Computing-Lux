/*
* Author Oleksandr Borysov
* Task1
*/

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

#define SEED 34237

#define MAX 32767

#define PLOT_DATA_FILE "plotData_1.txt"

__global__ void getPI(double* result, unsigned long* steps) {
	double x, y, z;
	long count = 0;

	curandState_t state;
	curand_init(SEED, 0, 0, &state);

	for (unsigned long i = 0; i < *steps; ++i) {
		x = ((double) ((curand(&state)) % MAX)) / MAX;
		y = ((double) ((curand(&state)) % MAX)) / MAX;
		z = sqrt((x * x) + (y * y));
		if (z <= 1) {
			++count;
		}
	}
	*result = ((double) count / *steps) * 4.0;
}

int main()
{
	cudaError_t cudaStatus;
	double* d_result;
	double result;
	unsigned long* d_stepNumber;
	unsigned long stepNumber;
	clock_t begin = clock();

	printf("Type number of steps \n");
	scanf("%d", &stepNumber);
	// Allocate memory in GPU
	cudaStatus = cudaMalloc(&d_result, sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_result failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc(&d_stepNumber, sizeof(long));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_stepNumber failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(d_stepNumber, &stepNumber, sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_stepNumber failed!");
        goto Error;
    }
	// Call cuda method
	getPI<<<1, 1>>>(d_result, d_stepNumber);

	 // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// Chek result
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getPI failed!");
        return 1;
    }

	cudaStatus = cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result failed!");
        goto Error;
    }
	printf("Calculated PI is = %f.\n", result);

Error:
	cudaFree(d_result);
	cudaFree(d_stepNumber);
    
	if (cudaStatus == 0) {
		double time_spent = (double) (clock() - begin) / CLOCKS_PER_SEC;
		FILE* dataPlotFile;
		dataPlotFile = fopen(PLOT_DATA_FILE, "a");
		fprintf(dataPlotFile, "%d %f\n", stepNumber, time_spent);
		fclose(dataPlotFile);
		printf("%d %f\n", stepNumber, time_spent);
	}
    return cudaStatus;
}
	
