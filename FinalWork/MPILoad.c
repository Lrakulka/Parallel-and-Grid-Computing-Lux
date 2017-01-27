/*
* Author Oleksandr Borysov
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <mpi.h>

#define MAX 32767

void getCounts(unsigned long* result, unsigned long* steps) {
	double x, y, z;
	unsigned int count = 0;
	srand((unsigned)time(NULL));

	for (unsigned long i = 0; i < *steps; ++i) {
		x = ((double)((rand()) % MAX)) / MAX;
		y = ((double)((rand()) % MAX)) / MAX;
		z = sqrt((x * x) + (y * y));
		if (z <= 1) {
			++count;
		}
	}
	*result = count;
}

int main(int argc, char* argv[]) {
	unsigned int stepNumber;
	int myid, numprocs;
	unsigned long procStep, result, count;
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
	
	getCounts(&count, &stepNumber);

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
		time_t tim = time(0);   // get time now
		struct tm now;
		localtime_s(&now, &tim);
		dataPlotFile = fopen("mpiLog_" + std::to_string(now.tm_mday) + "^"
			+ std::to_string(now.tm_hour) + "^" + std::to_string(now.tm_min) + ".txt", "a");
		fprintf(dataPlotFile, "%d %f %f %f %d\n", numprocs, maxTime, avrTime, minTime, steps);
		fclose(dataPlotFile);
	}
	MPI_Finalize();
}