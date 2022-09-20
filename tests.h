#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.141592653589793238462643383279502884L

#if TEST_NUMBER == 11

double a = 0.0;
double b = 1.0;


__device__ double kerr(double x, double t)
{
	return 0.5 * x * exp(t);
}


__device__ double func(double x)
{
	return exp(-x);
}


double ansolution(double x)
{
	return x + exp(-x);
}

#elif TEST_NUMBER == 12

double a = 1.0e-5;
double b = 0.5;


__device__ double kerr(double x, double t)
{
	return sin(x * t);
}


__device__ double func(double x)
{
	return 1.0 + (cos(0.5 * x) - 1.0) / x;
}


double ansolution(double x)
{
	return 1.0;
}

#elif TEST_NUMBER == 13

double a = 0.0;
double b = 2.0 * PI;


__device__ double kerr(double x, double t)
{
	return -1.0 / (pow(sin(0.5 * (x + t)), 2.0) + 0.25 * pow(cos(0.5 * (x + t)), 2.0)) / (4.0 * PI);
}


__device__ double func(double x)
{
	return (5.0 + 3.0 * cos(2.0 * x)) / (16.0 * PI);
}


double ansolution(double x)
{
	return (25.0 + 27.0 * cos(2.0 * x)) / (160.0 * PI);
}

#elif TEST_NUMBER == 14

double a = 0.0;
double b = 1.0;
#define LAMBDA 0.1


__device__ double kerr(double x, double t)
{
	return LAMBDA * (2.0 * x - t);
}


__device__ double func(double x)
{
	return x / 6.0;
}


double ansolution(double x)
{
	return (x + ((6.0 * x - 2.0) * LAMBDA - LAMBDA * LAMBDA) / (LAMBDA * LAMBDA - 3.0 * LAMBDA + 6.0)) / 6.0;
}

#elif TEST_NUMBER == 15

double a = 0.0;
double b = 2.0 * PI;

__device__ double kerr(double x, double t)
{
	return sin(x) * cos(t);
}


__device__ double func(double x)
{
	return cos(2.0 * x);
}


double ansolution(double x)
{
	return cos(2.0 * x);
}

#elif TEST_NUMBER == 16

double a = 0.0;
double b = 1.0;
#define LAMBDA 0.1


__device__ double kerr(double x, double t)
{
	return LAMBDA * (4.0 * x * t - x * x);
}


__device__ double func(double x)
{
	return x;
}


double ansolution(double x)
{
	return 3.0 * x * (2.0 * LAMBDA - 3.0 * LAMBDA * x + 6.0) / (LAMBDA * LAMBDA - 18.0 * LAMBDA + 18.0);
}

#elif TEST_NUMBER == 17

double a = 0.0;
double b = 1.0;


__device__ double kerr(double x, double t)
{
	return x * t * t;
}


__device__ double func(double x)
{
	return 1.0;
}


double ansolution(double x)
{
	return 1.0 + 4.0 * x / 9.0;
}

#elif TEST_NUMBER == 18

double a = 0.0;
double b = 1.0;


__device__ double kerr(double x, double t)
{
	return 0.5 * x * t;
}


__device__ double func(double x)
{
	return 5.0 * x / 6.0;
}


double ansolution(double x)
{
	return x;
}

#elif TEST_NUMBER == 19

double a = -1.0;
double b =  1.0;

__device__ double kerr(double x, double t)
{
	return x * x * exp(x * t);
}


__device__ double func(double x)
{
	return 1.0 - x * (exp(x) - exp(-x));
}


double ansolution(double x)
{
	return 1.0;
}

#elif TEST_NUMBER == 20

double a = 0.0;
double b = PI;
#define LAMBDA 1.0e-5


__device__ double kerr(double x, double t)
{
	return LAMBDA * pow(cos(x), 2.0);
}


__device__ double func(double x)
{
	return 1.0;
}


double ansolution(double x)
{
	return 1.0 + 2.0 * LAMBDA / (2.0 - PI * LAMBDA) * pow(cos(x), 2.0);
}

#endif
