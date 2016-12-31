/*
* Author Oleksandr Borysov
* Task1
*/
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define SEED 34237

#define MAX 32767

__global__ void getPI(double* result, unsigned long* steps) {
	double x, y, z;
	unsigned long count = 0;

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
	double* d_result;
	double result;
	unsigned long* d_stepNumber;
	unsigned long stepNumber;
	double *count_d;
	double count;
	printf("Type number of steps \n");
	scanf("%d", &stepNumber);

	cudaMalloc(&d_result, sizeof(double));
	cudaMalloc(&d_stepNumber, sizeof(long));

	cudaMemcpy(d_result, &result, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stepNumber, &stepNumber, sizeof(long), cudaMemcpyHostToDevice);

	getPI<<<1, 1>>>(d_result, d_stepNumber);

	cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

	printf("Calculated PI is = %f.\n", result);
	cudaFree(d_result);
	cudaFree(d_stepNumber);
    return 0;
}