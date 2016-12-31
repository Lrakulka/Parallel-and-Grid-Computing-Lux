
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h> 
#include <mpi.h>

#define MAX 32767

__global__ void getCounts(unsigned int* result, unsigned int* steps, unsigned int* seed) {
	double x, y, z;
	unsigned int count = 0;

	curandState_t state;
	curand_init(*seed, 0, 0, &state);

	for (unsigned long i = 0; i < *steps; ++i) {
		x = ((double)((curand(&state)) % MAX)) / MAX;
		y = ((double)((curand(&state)) % MAX)) / MAX;
		z = sqrt((x * x) + (y * y));
		if (z <= 1) {
			++count;
		}
	}
	*result = count;
}

int main(int argc, char* argv[]) {
	unsigned int stepNumber, myid, numprocs, resault, count;
	unsigned int procStep, result, seed;
	double elapse_time = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (myid == 0) {
		printf("Type number of steps \n");
		scanf("%d", &stepNumber);
		procStep = stepNumber / numprocs;
	}
	MPI_Bcast(&procStep, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	elapse_time = -MPI_Wtime();


	unsigned int *d_procStep, *d_result, *d_seed;
	//----------------
	cudaMalloc(&d_result, sizeof(int));
	cudaMalloc(&d_procStep, sizeof(int));
	cudaMalloc(&d_seed, sizeof(int));
	seed = time(NULL);
	cudaMemcpy(d_seed, &seed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_procStep, &procStep, sizeof(int), cudaMemcpyHostToDevice);

	getCounts <<<1, 1>>>(d_result, d_stepNumber, d_seed);

	cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaFree(d_procStep);
	cudaFree(d_seed);
	//--------------
	elapse_time += MPI_Wtime();
	MPI_Reduce(&result, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (myid == 0) {
		printf("Calculated PI is = %f.\n Time= %f\n", ((double)count / (procStep * numprocs)) * 4.0, elapse_time);
	}
	MPI_Finalize();
}
