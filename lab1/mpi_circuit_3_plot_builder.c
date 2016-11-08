#include <stdlib.h>
#include <stdio.h>
#define NUM_POINTS 5
#define NUM_COMMANDS 2

int main()
{
   FILE *dataPlot;
   unsigned int p[255], i = 0;
   double minTime[255], maxTime[255], avrTime[255];
   dataPlot = fopen("plotData.txt", "r");
   while(!feof(pFile)) {
      fscanf(dataPlot, "%d %f %f %f", p[i], minTime[i], maxTime[i], avrTime[i]);
      i++;
   }
   fclose(fp);

    char * commandsForGnuplot[] = {"set title \"TITLEEEEE\"", "plot 'plotData.txt'"};
    double xvals[NUM_POINTS] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yvals[NUM_POINTS] = {5.0 ,3.0, 1.0, 3.0, 5.0};
    FILE * temp = fopen("data.temp", "w");
    for (int i=0; i < NUM_POINTS; i++)
    {
    fprintf(temp, "%lf %lf \n", xvals[i], yvals[i]); //Write the data to a temporary file
    }
    /*Opens an interface that one can use to send commands as if they were typing into the
     *     gnuplot command line.  "The -persistent" keeps the plot open even after your
     *     C program terminates.
     */
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");

    for (i=0; i < NUM_COMMANDS; i++)
    {
    fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
    }
    return 0;
}
