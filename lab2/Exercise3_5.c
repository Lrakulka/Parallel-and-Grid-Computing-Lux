/*
* Lab 2
* Author Olecsandr Borysov
* Date 11.08.2016
*/

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit  */
#include <stdarg.h>    
#include <mpi.h>
#include <math.h>
#include <time.h>

#define N_INTERVAL 1000

#define SEED 35791246
#define PI 3.141592653589793238462643


int id = 0; // MPI id for the current process (set global to be used in xprintf)

void xprintf(char *format, ...) {
	va_list args;
	va_start(args, format);
	printf("[Node %i] ", id);
	vprintf(format, args);
	fflush(stdout);
}

int main(int argc, char ** argv) {
	int p;
	double elapsed_time = 0.0;
	double my_pi;
	double e;
	double points[N_INTERVAL];
	double minTime, maxTime, avrTime;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (p < 2) {
		xprintf("Please enter at least 2 processes");
		return 0;
	}
	if (id == 0) {
          xprintf("Enter the accuracy: ");
          scanf("%lf",&e);
    	}
  	MPI_Bcast(&e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();

	// GROUPS
	MPI_Group world_group;
	MPI_Group workers;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	int ranks[1] = { p - 1 };
	MPI_Group_excl(world_group, 1, ranks, &workers);

	MPI_Comm work_comm;
	MPI_Comm_create(MPI_COMM_WORLD, workers, &work_comm);

	// if generator node
	if (id == p - 1) {
		int request_workers = 1;
		int number;
		srand(SEED);
		while (request_workers) {
			MPI_Status status;
			MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if (!number) request_workers = 0;
			else {
				for (int i = 0; i < N_INTERVAL; i++) 
					points[i] = ((double)rand()) / ((double)RAND_MAX);
				MPI_Send(&points, N_INTERVAL, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			}
		}
	}
	// if other nodes
	else {
		MPI_Bcast(&e, 1, MPI_DOUBLE, 0, work_comm);
		int p_in = 0, p_out = 0, global_p_in = 0, global_p_out = 0;
		int request_generator = 1;
		MPI_Send(&request_generator, 1, MPI_INT, p - 1, 0, MPI_COMM_WORLD);

		while (request_generator) {
			MPI_Recv(&points, N_INTERVAL, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (int i = 0; i < N_INTERVAL - 1; i = i + 2) {
				if ((points[i] * points[i] + points[i + 1] * points[i + 1]) <= 1.0f) p_in++;
				else p_out++;
			}
			MPI_Allreduce(&p_in, &global_p_in, 1, MPI_INT, MPI_SUM, work_comm);
			MPI_Allreduce(&p_out, &global_p_out, 1, MPI_INT, MPI_SUM, work_comm);

			if (e > ((double)1 / ((double)2 * sqrt(global_p_in + global_p_out)))) request_generator = 0;

			MPI_Send(&request_generator, 1, MPI_INT, p - 1, 0, MPI_COMM_WORLD);
		}

		if (id == 0) {
			my_pi = 4.0f * (double)p_in / (double)(p_out + p_in);
			xprintf("Calculated PI = %.17f, Error is %.17f\n", my_pi, fabs(my_pi - PI));
		}

	}

	// at the end, compute elapsed time
	elapsed_time += MPI_Wtime();
	MPI_Reduce(&elapsed_time, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &avrTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (id == 0) {
		avrTime /= p;
		FILE* dataPlotFile;
    		dataPlotFile = fopen("plotDataExc3_5.txt", "a");
	    	fprintf(dataPlotFile, "%d %f %f %f\n", p, maxTime, avrTime, minTime);
	    	fclose(dataPlotFile);
	}

	MPI_Finalize();
	return 0;
}
