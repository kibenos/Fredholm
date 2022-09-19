#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef TEST_NUMBER

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

#endif //!TEST_NUMBER
