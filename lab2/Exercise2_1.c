/**
* Author Oleksandr Borysov
* data 11.7.2016
* lab 2 Exercise 1
**/

#include <math.h>
#include <stdio.h>

const double PI = 3.141592653589793238462643;  // 25 elements of PI

double integrate_f(double);          /* Integral function */
double simpson(int, double, double, double, int);

int main(int argc, char *argv[]) {
  int n;                   
  double total; 
  double pi, y, x1, x2, l, sum;
  
  printf("Enter the number of intervals: ");
  scanf("%d",&n);

  /* Each processor computes its interval */
  x1 = 0;
  x2 = 1;
  
  l = 1 / ((double) (2 * n));
  sum = 0.0;
  for (int i = 1; i < n ; i++)
  {
    y = x1 + (x2 - x1) * i / ((double) n);
    sum = simpson(i, y, l, sum, n);
  }

  sum += (integrate_f(x1) + integrate_f(x2)) / 2.0;
  pi = sum * 2.0 * l/ 3.0;
  printf("Calculated pi %.16f\n Error of calculated pi is %.16f\n", pi, fabs(pi - PI));
  return 0;
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
