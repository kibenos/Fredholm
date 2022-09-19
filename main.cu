#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define TEST_NUMBER 19 // 11-20
#include "tests.h"
#include "config.h"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE2D 32
#define MIDX(i, j, ld) ((j) * ld + (i)) 
#define PI 3.141592653589793238462643383279502884L
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuGetLastError() gpuErrchk(cudaGetLastError())

inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << "[" << file << ":" << line << "] " << cudaGetErrorString(code) << "\n";
		exit(-1);
	}
}

inline void gpuAssert(cublasStatus_t code, const char* file, int line)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "[" << file << ":" << line << "] cuBLAS error " << code << ".\n";
		exit(-1);
	}
}

inline void gpuAssert(cusolverStatus_t code, const char* file, int line)
{
	if (code != CUSOLVER_STATUS_SUCCESS)
	{
		std::cout << "[" << file << ":" << line << "] cuSOLVER error " << code << ".\n";
		exit(-1);
	}
}


#define RULE RECTANGLE_RULE
//#define RULE TRAPEZOIDAL_RULE
//#define RULE SIMPSONS_RULE

extern double a;
extern double b;

namespace rule
{
	constexpr double eps  = 1.0e-7;
	constexpr size_t maxp = 2'000;

#if RULE == RECTANGLE_RULE

	constexpr unsigned int npoints           = 1;
	constexpr unsigned int nweights          = 1;
	constexpr double       weights[nweights] = { 1.0 };
	constexpr double       border_weight     = 1.0;
	constexpr double       sum_coeff         = 1.0;
	constexpr double       order             = 2.0;

	inline double totalnpoints(size_t nh) { return nh; }

#elif RULE == TRAPEZOIDAL_RULE

	constexpr unsigned int npoints           = 2;
	constexpr unsigned int nweights          = 1;
	constexpr double       weights[nweights] = { 1.0 };
	constexpr double       border_weight     = 0.5;
	constexpr double       sum_coeff         = 1.0;
	constexpr double       order             = 2.0;

	inline double totalnpoints(size_t nh) { return nh + 1; }

#else //SIMPSONS_RULE

	constexpr unsigned int npoints           = 3;
	constexpr unsigned int nweights          = 2;
	constexpr double       weights[nweights] = { 1.0, 2.0 };
	constexpr double       border_weight     = 0.5;
	constexpr double       sum_coeff         = (1.0/3.0);
	constexpr double       order             = 4.0;

	inline double totalnpoints(size_t nh) { return 2 * nh + 1; }

#endif

}


__constant__ __device__ unsigned int c_nweights;
__constant__ __device__ double       c_weights[rule::nweights];
__constant__ __device__ double       c_border_weight;


__device__ inline double weight(size_t i, size_t n)
{
	if (i > 0 && i + 1 < n)
		return c_weights[i % c_nweights];
	return c_border_weight;
}


struct mesh_t
{
	double* points;
	double  a, b, h;
	size_t  nh, npoints;

	mesh_t(double a, double b, size_t nh);
};


struct approxfunc_t
{
	mesh_t  mesh;
	double* values;

	approxfunc_t(double a, double b, size_t nh)
	:
		mesh(a, b, nh),
		values{nullptr}
	{
		gpuErrchk(cudaMalloc(&values, mesh.npoints * sizeof(double)));
	}
};


template<unsigned int NPOINTS>
__global__ void setPoints(mesh_t const mesh)
{
	size_t i = gridDim.x * blockIdx.x + threadIdx.x;

	if (NPOINTS == 1)
	{
		if (i < mesh.npoints)
			mesh.points[i] = mesh.a + mesh.h * (2 * i + 1) / 2.0;
	}
	else
	{
		if (i < mesh.npoints)
			mesh.points[i] = mesh.a + mesh.h * ((i / (NPOINTS - 1)) + (double)(i % (NPOINTS - 1)) / (NPOINTS - 1));

		if (i + 1 == mesh.npoints)
			mesh.points[i] = mesh.b;
	}
}


mesh_t::mesh_t(double a, double b, size_t nh)
{
	this->a  = a;
	this->b  = b;
	this->nh = nh;

	h = (b - a) / nh;
	npoints = rule::totalnpoints(nh);

	gpuErrchk(cudaMalloc(&points, npoints * sizeof(double)));
	setPoints<rule::npoints><<<(npoints + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
	gpuErrchk(cudaDeviceSynchronize());
}


__global__ void mulWeights(double* x, size_t n)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n)
		x[i] *= weight(i, n);
}


__global__ void wrapL2(double* x, size_t n)
{
	size_t const i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		x[i] = pow(x[i], 2.0);
}


__global__ void fill(double* x, double a, size_t n)
{
	size_t const i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		x[i] = a;
}


__global__ void getKerrApproxMatrix(mesh_t const mesh, double rulesumcoeff, double* matrix)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < mesh.npoints && j < mesh.npoints)
		matrix[MIDX(i, j, mesh.npoints)] = 1.0 * (i == j) - rulesumcoeff * mesh.h * weight(j, mesh.npoints) * kerr(mesh.points[i], mesh.points[j]);
}


__global__ void getRhs(mesh_t const mesh, double* rhs)
{
	size_t i = gridDim.x * blockIdx.x + threadIdx.x;

	if (i < mesh.npoints)
		rhs[i] = func(mesh.points[i]);
}


void solveNxNsystem(cusolverDnHandle_t handle,
	double*  matrix,
	double*  rhs,
	double** workspace,
	size_t*  workspace_size,
	size_t   n
)
{
	size_t const ld = n;

	int Lwork;
	gpuErrchk(cusolverDnDgetrf_bufferSize(handle, n, n, matrix, ld, &Lwork));

	if (Lwork > *workspace_size || *workspace == nullptr)
	{
		*workspace_size = Lwork;
		gpuErrchk(cudaFree(*workspace));
		gpuErrchk(cudaMalloc(workspace, Lwork));
	}

	int* p_dev;
	gpuErrchk(cudaMalloc(&p_dev, n * sizeof(int)));

	int* info_dev;
	gpuErrchk(cudaMalloc(&info_dev, sizeof(int)));

	gpuErrchk(cusolverDnDgetrf(handle, n, n, matrix, ld, *workspace, p_dev, info_dev));

	int info = 0;
	gpuErrchk(cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
	if (info > 0)
	{
		std::cout << "error in cusolverDnDgetrf: matrix is singular.\n";
		exit(EXIT_FAILURE);
	}
	if (info < 0)
	{
		std::cout << "error in cusolverDnDgetrf: invalid " << (-info) << "th argument.\n";
		exit(EXIT_FAILURE);
	}

	gpuErrchk(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, matrix, ld, p_dev, rhs, n, info_dev));

	info = 0;
	gpuErrchk(cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
	if (info > 0)
	{
		std::cout << "error in cusolverDnDgetrs: matrix is singular.\n";
		exit(EXIT_FAILURE);
	}
	if (info < 0)
	{
		std::cout << "error in cusolverDnDgetrs: invalid " << (-info) << "th argument.\n";
		exit(EXIT_FAILURE);
	}

	gpuErrchk(cudaFree(info_dev));
	gpuErrchk(cudaFree(p_dev));
}


__global__ void approxSolutionOnNewMesh_sum(
	mesh_t const       mewmesh,
	approxfunc_t const solution,
	double*            sum
)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < mewmesh.npoints && j < solution.mesh.npoints)
		sum[MIDX(i, j, mewmesh.npoints)] =
			weight(j, solution.mesh.npoints) * 
			kerr(mewmesh.points[i], solution.mesh.points[j]) * solution.values[j];
}


// device mem: sizeof(workspace) = solution.mesh.npoints * newsolution.mesh.npoints
// device mem: sizeof(ones) = solution.mesh.npoints, filled with 1.0
void approxSolutionOnNewMesh(cublasHandle_t handle,
	approxfunc_t const solution,
	approxfunc_t const newsolution,
	double const*      ones,
	double*            workspace
)
{
	size_t m = newsolution.mesh.npoints;
	size_t n = solution.mesh.npoints;

	dim3 grid = {
		(unsigned int)((m + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
		(unsigned int)((n + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
		1
	};

	dim3 block = { BLOCK_SIZE2D, BLOCK_SIZE2D, 1 };

	approxSolutionOnNewMesh_sum<<<grid, block>>>(newsolution.mesh, solution, workspace);
	gpuErrchk(cudaDeviceSynchronize());

	getRhs<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(newsolution.mesh, newsolution.values);
	gpuErrchk(cudaDeviceSynchronize());

	double alpha = rule::sum_coeff * solution.mesh.h;
	double beta  = 1.0;

	gpuErrchk(cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, workspace, m, ones, 1, &beta, newsolution.values, 1));
}


// device mem: sizeof(workspace) = newmesh.npoints * (max(solution1.mesh.npoints, solution2.mesh.npoints) + 1)
// device mem: sizeof(ones) = max(solution1.mesh.npoints, solution2.mesh.npoints, newmesh.npoints), filled with 1.0
double getSolutionsDiffL2onNewMesh(cublasHandle_t handle,
	approxfunc_t const solution1,
	approxfunc_t const solution2,
	mesh_t const       newmesh,
	double const*      ones,
	double*            workspace
)
{
	size_t m = newmesh.npoints;
	size_t n = std::max({ solution1.mesh.npoints, solution2.mesh.npoints });

	dim3 grid = {
		(unsigned int)((m + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
		(unsigned int)((n + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
		1
	};

	dim3 block = { BLOCK_SIZE2D, BLOCK_SIZE2D, 1 };

	gpuErrchk(cudaMemset(workspace, 0, m * n * sizeof(double)));
	approxSolutionOnNewMesh_sum<<<grid, block>>>(newmesh, solution1, workspace);
	gpuErrchk(cudaDeviceSynchronize());

	double* newvalues = workspace + m * n;
	double  alpha     = solution1.mesh.h;
	double  beta      = 0.0;

	gpuErrchk(cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, workspace, m, ones, 1, &beta, newvalues, 1));

	gpuErrchk(cudaMemset(workspace, 0, m * n * sizeof(double)));
	approxSolutionOnNewMesh_sum<<<grid, block>>>(newmesh, solution2, workspace);
	gpuErrchk(cudaDeviceSynchronize());

	alpha = solution2.mesh.h;
	beta  = -1.0;

	gpuErrchk(cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, workspace, m, ones, 1, &beta, newvalues, 1));

	wrapL2<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(newvalues, m);
	gpuErrchk(cudaDeviceSynchronize());

	mulWeights<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(newvalues, m);
	gpuErrchk(cudaDeviceSynchronize());

	alpha = 1.0;
	beta  = 0.0;
	double sum = 0.0;

	gpuErrchk(cublasDgemv(handle, CUBLAS_OP_N, 1, m, &alpha, newvalues, 1, ones, 1, &beta, workspace, 1));
	gpuErrchk(cudaMemcpy(&sum, workspace, sizeof(double), cudaMemcpyDeviceToHost));

	return rule::sum_coeff * rule::sum_coeff * rule::sum_coeff * newmesh.h * sum;
}


// device mem: sizeof(workspace) = nhscale * rule::npoints * (max(solution1.mesh.npoints, solution2.mesh.npoints) + 1)
// device mem: sizeof(ones) = max(solution1.mesh.npoints, solution2.mesh.npoints), filled with 1.0
double checkConvergence(cublasHandle_t handle,
	approxfunc_t const solution1,
	approxfunc_t const solution2,
	double const       h0,
	size_t const       nhscale,
	double const*      ones,
	double*            workspace
)
{
	double const a = std::min({ solution1.mesh.a, solution2.mesh.a });
	double const b = std::max({ solution1.mesh.b, solution2.mesh.b });
	double const lambda  = 1.0 / nhscale;

	double sumab    = 0.0;
	double sumhprev = 0.0;
	double sumhcurr = 0.0;
	double acurr    = a;
	double hcurr    = h0;
	size_t nhcurr   = 1;
	double errorh   = 0.0;

	for (size_t i = 0; abs(b - acurr) > rule::eps; i++)
	{
		mesh_t mesh(acurr, acurr + hcurr, nhcurr);

		sumhprev = sumhcurr;
		sumhcurr = getSolutionsDiffL2onNewMesh(handle, solution1, solution2, mesh, ones, workspace);

		errorh = abs(sumhcurr - sumhprev) * hcurr / (b - a) / (1.0 - pow(2.0, -(double)rule::order));

		if ((i + 1) % 1'000 == 0)
			std::cout << "adaptive algorithm. iteration: " << (i + 1) << " (" << ((acurr - a) / (b - a)) << "%)\n";

		if (i && errorh < rule::eps)
		{
			acurr += hcurr;
			sumab += sumhcurr;
			hcurr  = h0;

			if (acurr + hcurr > b)
				hcurr = b - acurr;
		}
		else if (nhcurr == nhscale)
		{
			nhcurr = 1;
			hcurr *= lambda;
		}
		else if (nhcurr == 1)
		{
			nhcurr = nhscale;
		}

		gpuErrchk(cudaFree(mesh.points));
	}

	return sqrt(sumab);
}


int main(int argc, char** argv)
{
	size_t nh                 = 20;
	size_t const nhxscale     = 1;
	size_t const nhpscale     = 50;
	double const hadapt       = 0.5;
	size_t const nhscaleadapt = 2;
	size_t const anN          = 500;

	double* approxmatrix = nullptr;
	double* ones         = nullptr;
	double* workspace    = nullptr;
	size_t workspacesize = 0;

	approxfunc_t* currsolution = nullptr, *prevsolution = nullptr;
	double error = 0.0;

	dim3 const block = { BLOCK_SIZE2D, BLOCK_SIZE2D, 1 };

	gpuErrchk(cudaMemcpyToSymbol(c_nweights, &rule::nweights, sizeof(unsigned int)));
	gpuErrchk(cudaMemcpyToSymbol(c_weights, &rule::weights, rule::nweights * sizeof(double)));
	gpuErrchk(cudaMemcpyToSymbol(c_border_weight, &rule::border_weight, sizeof(double)));

	cublasHandle_t cublas_handle;
	gpuErrchk(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle;
	gpuErrchk(cusolverDnCreate(&cusolver_handle));

	size_t n = 0;
	for (int i = 0; ; i++)
	{
		currsolution = new approxfunc_t(a, b, nh);
		n = currsolution->mesh.npoints;

		dim3 grid = {
			(unsigned int)((n + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
			(unsigned int)((n + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D),
			1
		};

		gpuErrchk(cudaFree(ones));
		gpuErrchk(cudaMalloc(&ones, n * sizeof(double)));
		fill<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(ones, 1.0, n);
		gpuErrchk(cudaDeviceSynchronize());

		workspacesize = (n + 1) * nhscaleadapt * rule::npoints * sizeof(double);
		gpuErrchk(cudaFree(workspace));
		gpuErrchk(cudaMalloc(&workspace, workspacesize));

		gpuErrchk(cudaFree(approxmatrix));
		gpuErrchk(cudaMalloc(&approxmatrix, n * n * sizeof(double)));
		getKerrApproxMatrix<<<grid, block>>>(currsolution->mesh, rule::sum_coeff, approxmatrix);
		gpuErrchk(cudaDeviceSynchronize());

		getRhs<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(currsolution->mesh, currsolution->values);
		gpuErrchk(cudaDeviceSynchronize());

		solveNxNsystem(cusolver_handle, approxmatrix, currsolution->values, &workspace, &workspacesize, n);

		if (i)
		{
			error = checkConvergence(cublas_handle, *currsolution, *prevsolution, hadapt, nhscaleadapt, ones, workspace);

			if ((i + 1) % 1 == 0)
				std::cout << "iteration: " << (i + 1) << "; h = " << currsolution->mesh.h << "; error = " << error << "; n = " << n << "\n";

			if (error < rule::eps || n > rule::maxp)
			{
				gpuErrchk(cudaFree(prevsolution->values));
				gpuErrchk(cudaFree(prevsolution->mesh.points));
				delete prevsolution;
				break;
			}
		}

		if (prevsolution)
		{
			gpuErrchk(cudaFree(prevsolution->values));
			gpuErrchk(cudaFree(prevsolution->mesh.points));
			delete prevsolution;
		}

		prevsolution = currsolution;
		currsolution = nullptr;

		nh = nh * nhxscale + nhpscale;
	}

	gpuErrchk(cudaFree(approxmatrix));
	gpuErrchk(cudaFree(ones));
	gpuErrchk(cudaFree(workspace));

	if (!currsolution)
		return EXIT_FAILURE;

	double* solp = new double[n];
	double* solv = new double[n];
	double* sol  = new double[2 * n];
	gpuErrchk(cudaMemcpy(solp, currsolution->mesh.points, n * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(solv, currsolution->values, n * sizeof(double), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 2 * n; i++)
	{
		if (i % 2 == 0)
		{
			sol[i]   = solp[i / 2];
			//std::cout << "(" << sol[i] << "; ";
		}
		else
		{
			sol[i]   = solv[(i - 1) / 2];
			//std::cout << sol[i] << ")\n";
		}
	}
	//std::cout << "\n";

	std::ofstream out("nusolution.dat", std::ios::out | std::ios::binary);
	out.write((char *)sol, 2 * n * sizeof(double));
	out.close();

	delete[] solp;
	delete[] solv;
	delete[] sol;

	double* ansol = new double[2 * anN];
	for (int i = 0; i < 2 * anN; i++)
	{
		if (i % 2 == 0)
		{
			ansol[i] = a + (b - a) / anN * (i / 2);

			if (i == 2 * anN - 2)
				ansol[i] = b;
		}
		else
		{
			ansol[i] = ansolution(ansol[i - 1]);
		}
	}

	out.open("ansolution.dat", std::ios::out | std::ios::binary);
	out.write((char*)ansol, 2 * anN * sizeof(double));
	out.close();

	delete[] ansol;

	gpuErrchk(cudaFree(currsolution->values));
	gpuErrchk(cudaFree(currsolution->mesh.points));
	delete currsolution;

	std::string command = "gnuplot -persist -e \"eps = " + std::to_string(rule::eps) + "; \" draw_solution.gpl";
	system(command.c_str());

	return EXIT_SUCCESS;
}