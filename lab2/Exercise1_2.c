/**
* Author Oleksandr Borysov
* data 11.7.2016
* lab 2 Exercise 1
**/

#include <math.h>
#include <stdio.h>

const double PI = 3.141592653589793238462643;  // 25 elements of PI

int main(int argc, char *argv[])
{
    int n, numprocs;
    double mypi, h, sum, x;
    printf("Enter the number of intervals: ");
    scanf("%d",&n);
    h = 1.0 / n;
    sum = 0.0;
    for (int i = 1; i <= n; i++) {
       x = h * ((double) i - 0.5);
       sum += 4.0 / (1.0 + x * x);
    }
    mypi = h * sum;
    printf("Calculated pi %.16f\n Error of calculated pi is %.16f\n", mypi, fabs(mypi - PI));
    return 0;
}
