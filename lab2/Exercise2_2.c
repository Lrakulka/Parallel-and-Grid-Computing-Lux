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

double integrate_f(double);          /* Integral function */
double simpson(int, double, double, double, int);

/**
 * Redefinition of the xprintf to include the buffer flushing
 */
void xprintf(char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[Node %i] ", myid);
    vprintf(format, args);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
  int numprocs, n;
  double pi, y, processor_output_share[32], x1, x2, l, sum;
  MPI_Status status;
  double minTime, maxTime, avrTime;
  double elapse_time = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (myid == 0) {
       xprintf("Enter the number of intervals: ");
       scanf("%d",&n);
    }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  elapse_time = -MPI_Wtime();

  /* Each processor computes its interval */
  x1 = myid / ((double) numprocs);
  x2 = (myid + 1) / ((double) numprocs);
  
  /* l is the same for all processes. */
  l = 1 / ((double) (2 * n * numprocs));
  sum = 0.0;
  for (int i = 1; i < n ; i++)
  {
    y = x1 + (x2 - x1) * i / ((double) n);
    sum = simpson(i, y, l, sum, n);
  }

  /* Include the endpoints of the intervals */
  sum += (integrate_f(x1) + integrate_f(x2))/2.0;

  elapse_time += MPI_Wtime();
  
  MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&elapse_time, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapse_time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapse_time, &avrTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    pi *= 2.0 * l/ 3.0;
    xprintf("Calculated pi %.16f\n Error of calculated pi is %.16f\n", pi, fabs(pi - PI));

    avrTime /= numprocs;
    xprintf("done! \n Found max process time %f\n Found avr process time %f\n Found min process time %f\n", maxTime, avrTime, minTime);
    FILE* dataPlotFile;
    dataPlotFile = fopen("plotDataExc2_2,3.txt", "a");
    fprintf(dataPlotFile, "%d %f %f %f\n", numprocs, maxTime, avrTime, minTime);
    fclose(dataPlotFile);
  }
  
  MPI_Finalize();
}
 

double integrate_f(double x) {
  return 4.0/(1.0 + x * x);     
}

double simpson(int i, double y, double l, double sum, int n) {
  sum += integrate_f(y);
  sum += 2.0 * integrate_f(y - l);
  if(i == (n - 1))
    sum += 2.0 * integrate_f(y + l);
  return sum;
}
