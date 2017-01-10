/*
* Author Oleksandr Borysov
* Task2
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h> 
#include <mpi.h>

#define MAX 32767
#define PLOT_DATA_FILE "plot_data2.txt"

__global__ void getCounts(unsigned long* result, unsigned long* steps, unsigned long* seed) {
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
	unsigned int stepNumber;
	int myid, numprocs, dev_used;
	unsigned long procStep, result, seed, resault, count;
	cudaError_t cudaStatus;
	double minTime, maxTime, avrTime, elapse_time = 0.0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (myid == 0) {
		printf("Type number of steps \n");
		scanf("%u", &stepNumber);
		procStep = stepNumber / numprocs;
	}
	MPI_Bcast(&procStep, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	elapse_time = -MPI_Wtime();
	cudaSetDevice(myid);
	cudaGetDevice(&dev_used); // Find which GPU is being used
    printf("myid = %d: device used = %d\n", myid, dev_used);

	unsigned long *d_procStep, *d_result, *d_seed;
	//----------------
	cudaStatus = cudaMalloc(&d_result, sizeof(long));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_result failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc(&d_procStep, sizeof(long));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_procStep failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc(&d_seed, sizeof(long));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_seed failed!");
        goto Error;
    }
	// Genereterandom seed
	seed = time(NULL);

	cudaStatus = cudaMemcpy(d_seed, &seed, sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_seed failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(d_procStep, &procStep, sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_procStep failed!");
        goto Error;
    }
	// run CUDA method
	getCounts <<<1, 1>>>(d_result, d_procStep, d_seed);

	cudaStatus = cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_result failed!");
        goto Error;
    }

	//--------------
	MPI_Reduce(&result, &count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	elapse_time += MPI_Wtime();
    MPI_Reduce(&elapse_time, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapse_time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapse_time, &avrTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
Error:
	cudaFree(d_result);
	cudaFree(d_procStep);
	cudaFree(d_seed);

	if (myid == 0 && cudaStatus == 0) {
		int steps = procStep * numprocs;
		printf("Calculated PI is = %f.\n Time= %f\n", ((double)count / steps) * 4.0, elapse_time);
		avrTime /= numprocs;
		FILE* dataPlotFile;
		dataPlotFile = fopen(PLOT_DATA_FILE, "a");
		fprintf(dataPlotFile, "%d %f %f %f %d\n", numprocs, maxTime, avrTime, minTime, steps);
		fclose(dataPlotFile);
	}
	MPI_Finalize();
}