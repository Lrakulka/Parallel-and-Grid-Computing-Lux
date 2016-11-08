/**
* Author Oleksandr Borysov
* data 11.7.2016
* lab 2 Exercise 1
**/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

int myid; // Id of the process
const double PI = 3.141592653589793238462643;  // 25 elements of PI

/**
 * Redefinition of the printf to include the buffer flushing
 */
void xprintf(char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[Node %i] ", myid);
    vprintf(format, args);
    fflush(stdout);
}


int main(int argc, char *argv[])
{
    int n, numprocs;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    if (myid == 0) {
       xprintf("Enter the number of intervals: ");
       scanf("%d",&n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / (double) n;
    sum = 0.0;
    for (int i = myid + 1; i <= n; i += numprocs) {
       x = h * ((double) i - 0.5);
       sum += 4.0 / (1.0 + x * x);
    }
    mypi = h * sum;

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {
       FILE* dataPlotFile;
       dataPlotFile = fopen("plotDataExc1_3.txt", "a");
       fprintf(dataPlotFile, "%d %f \n", numprocs, fabs(pi - PI));
       fclose(dataPlotFile);

       xprintf("Calculated pi %.16f\n Error of calculated pi is %.16f\n", pi, fabs(pi - PI));
    }
    MPI_Finalize();
    return 0;
}