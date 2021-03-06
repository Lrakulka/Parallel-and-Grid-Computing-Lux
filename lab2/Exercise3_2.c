/*
* Lab 2
* Author Olecsandr Borysov
* Date 11.08.2016
*/

#include <stdio.h>
#include <stdarg.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RAND_VALUE 763453

const double PI = 3.141592653589793238462643;  // 25 elements of PI

int main (int argc, char *argv[]) {
	int n = 0;
	double x, y;
	double z, myPi;
	int count = 0;
	double e;
	FILE *in, *out;

	printf("Enter the accuracy: ");
        scanf("%lf",&e);

	in = fopen("dataPlotIn.txt", "a");
	out = fopen("dataPlotOut.txt", "a");
	srand(RAND_VALUE); 
	do {
		n++;
		x = rand() / ((double)RAND_MAX);
		y = rand() / ((double)RAND_MAX);
		z = sqrt((x * x) + (y * y));
		if (z <= 1) {
			count++;
			fprintf(in, "%f %f\n", x, y);
		} else {
			fprintf(out, "%f %f\n", x, y);
		}
	} while (e < ( 1 / ((double)2 * sqrt(n))));
	myPi = (count / (double) n) * 4.0;
	fclose(in);
	fclose(out);
	printf("Calculated pi %.16f\n Error of calculated pi is %.16f\n", myPi, fabs(myPi - PI));
}
