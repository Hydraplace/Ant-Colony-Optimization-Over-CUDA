#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#define Infinity 65536
#define index(x,y,z) (z+y*x)
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}
using namespace std;
const int n_iterations = 300;
const double initial_pheromone = 1.0;
const double evap_rate = 0.5;
const double ALFA = 1;
const double BETA = 2;
const double GAMA = 2;
int *load_adjacency_matrix(char const *filename, int &n_cities);
int calculate_tourcost(int *distances, int *costs,int *path, int n_cities);
int calculate_tourcost_single(int *distances,int *path, int n_cities);
int *optimal_solution(int *tours, int *distances,int *costs, int n_ants, int n_cities);
void evaporate(double *pheromones, int n_cities);
void pheromone_update(double *pheromones, int *distances, int *costs,int*min_path, int n_cities);
int *aco_cuda(int *distances, int *costs,int n_cities, int n_ants);
__global__ void cuda_evaporate(double *pheromones, int n_cities, double evap_rate);
__global__ void cuda_pheromone_update(double *pheromones, int *distances, int*costs,int *path, int n_cities, double amount);
__global__ void cuda_path_traverse(int *tours, int *visited, double *choiceinfo, double *probs, int n_cities);

int main()
{
	srand((unsigned)time(NULL));
	char const *inputfile, *outputfile,*cinputfile;
	inputfile = "distance.txt";
	outputfile = "Output.txt";
	cinputfile = "cost.txt";
	int n_cities;
	int *distances;
	int *costs;
	distances = load_adjacency_matrix(inputfile, n_cities);
	costs= load_adjacency_matrix(cinputfile, n_cities);
	int *solution = aco_cuda(distances, costs, n_cities, n_cities);
	int cost = calculate_tourcost_single(distances, solution, n_cities);
	int cost2 = calculate_tourcost_single(costs, solution, n_cities);
	ofstream output;
	output.open(outputfile);
	output << "Total cost of traversal: " << cost <<"\t"<<"\nTotal expenditure: "<<cost2<<endl;
	output << "Best Solution Path:\n";
	for (int i = 0; i < n_cities; i++)
		output << solution[i] << endl;
	output << solution[0] << endl;
	cout << "CUDA ACO is complete" << endl;
	return 0;
}

__global__ void cuda_evaporate(double *pheromones, int n_cities, double evap_rate)
{
	int edge_id = threadIdx.x + blockIdx.x*blockDim.x;
	pheromones[edge_id] *= evap_rate;
}

__global__ void cuda_pheromone_update(double *pheromones, int *distances,int *cost, int *path, int n_cities, double amount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int first = path[id];
	int second = path[id + 1];
	pheromones[index(n_cities, first, second)] += amount;
	pheromones[index(n_cities, second, first)] += amount;
}

__global__ void cuda_path_traverse(int *tours, int *visited, double *choiceinfo, double *probs, int n_cities)
{
	int line_id = blockDim.x*blockIdx.x + threadIdx.x;
	for (int step = 1; step < n_cities; step++)
	{
		int current = tours[index(n_cities, line_id, step - 1)];
		double total_prob = 0.0;
		for (int i = 0; i < n_cities; i++)
		{
			if (visited[index(n_cities, line_id, i)] == 1)
				probs[index(n_cities, line_id, i)] = 0.0;
			else {
				double current_prob = choiceinfo[index(n_cities, current, i)];
				probs[index(n_cities, line_id, i)] = current_prob;
				total_prob += current_prob;
			}
		}
		double random;
		curandState_t state;
		curand_init((unsigned long long) clock(), 0, 0, &state);
		random = curand_uniform(&state);
		random *= total_prob;
		int next;
		double sum = probs[index(n_cities, line_id, 0)];
		for (next = 0; sum < random; next++)
		{
			sum += probs[index(n_cities, line_id, next + 1)];
		}
		tours[index(n_cities, line_id, step)] = next;
		visited[index(n_cities, line_id, next)] = 1;
	}
}


int *load_adjacency_matrix(char const *filename, int &n_cities)
{
	ifstream adj_matrix;
	adj_matrix.open(filename);
	adj_matrix >> n_cities;
	int* distances = (int *)malloc(n_cities*n_cities * sizeof(int));
	for (int i = 0; i < n_cities; i++)
		for (int j = 0; j < n_cities; j++)
			adj_matrix >> distances[index(n_cities, i, j)];
	return distances;
}

int calculate_tourcost(int *distances, int *costs,int *path, int n_cities)
{
	int cost = 0;
	for (int i = 0; i < (n_cities - 1); i++)
	{
		cost += distances[index(n_cities, path[i], path[i + 1])];
		cost += costs[index(n_cities, path[i], path[i + 1])];
	}
	cost += distances[index(n_cities, path[n_cities - 1], path[0])];
	cost += costs[index(n_cities, path[n_cities - 1], path[0])];
	return cost;
}
int calculate_tourcost_single(int *distances, int *path, int n_cities)
{
	int cost = 0;
	for (int i = 0; i < (n_cities - 1); i++)
	{
		cost += distances[index(n_cities, path[i], path[i + 1])];
	}
	cost += distances[index(n_cities, path[n_cities-1], path[0])];
	return cost;
}

int *optimal_solution(int *tours, int *distances,int *costs, int n_ants, int n_cities)
{
	int *best_tour = &tours[0];
	for (int tour = 0; tour < n_ants; tour++)
		if (calculate_tourcost(distances,costs, &tours[index(n_cities, tour, 0)], n_cities) < calculate_tourcost(distances,costs, best_tour, n_cities))
			best_tour = &tours[index(n_cities, tour, 0)];
	return best_tour;
}

void evaporate(double *pheromones, int n_cities)
{
	int size = n_cities * n_cities * sizeof(double);
	double *pheromones_device;
	CudaSafeCall(cudaMalloc((void**)&pheromones_device, size));
	cudaMemcpy(pheromones_device, pheromones, size, cudaMemcpyHostToDevice);
	cuda_evaporate << < n_cities, n_cities >> > (pheromones_device, n_cities, evap_rate);
	CudaCheckError();
	cudaMemcpy(pheromones, pheromones_device, size, cudaMemcpyDeviceToHost);
	cudaFree(pheromones_device);
}

void pheromone_update(double *pheromones, int *distances,int *costs, int *path, int n_cities)
{
	double amount = (double)(((double)(1.0f / (double)calculate_tourcost_single(distances,path, n_cities)))+((double)(1.0f / (double)calculate_tourcost_single(distances, path, n_cities))));
	int size_path = n_cities * sizeof(int);
	int size_int = n_cities * n_cities * sizeof(int);
	int size_double = n_cities * n_cities * sizeof(double);
	int *path_device;
	int *distances_device;
	int *costs_device;
	double *pheromones_device;
	CudaSafeCall(cudaMalloc((void**)&path_device, size_path));
	CudaSafeCall(cudaMalloc((void**)&distances_device, size_int));
	CudaSafeCall(cudaMalloc((void**)&costs_device, size_int));
	CudaSafeCall(cudaMalloc((void**)&pheromones_device, size_double));
	cudaMemcpy(path_device, path, size_path, cudaMemcpyHostToDevice);
	cudaMemcpy(distances_device, distances, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(costs_device, costs, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(pheromones_device, pheromones, size_double, cudaMemcpyHostToDevice);
	cuda_pheromone_update << < 1, n_cities - 1 >> > (pheromones_device, distances_device,costs_device, path_device, n_cities, amount);
	CudaCheckError();
	cudaMemcpy(distances, distances_device, size_int, cudaMemcpyDeviceToHost);
	cudaMemcpy(costs, costs_device, size_int, cudaMemcpyDeviceToHost);
	cudaMemcpy(pheromones, pheromones_device, size_double, cudaMemcpyDeviceToHost);
	cudaFree(path_device);
	cudaFree(distances_device);
	cudaFree(pheromones_device);
}

int *aco_cuda(int *distances, int *costs, int n_cities, int n_ants)
{
	int ph_size = n_cities * n_cities * sizeof(double);
	int tours_size = n_ants * n_cities * sizeof(int);
	int dist_size = n_cities * n_cities * sizeof(int);
	double *pheromones = (double*)malloc(ph_size);
	int *tours = (int*)malloc(tours_size);
	int *visited = (int*)malloc(tours_size);
	double *choiceinfo = (double*)malloc(ph_size);
	int *distances_device;
	int *costs_device;
	int *tours_device;
	int *visited_device;
	double *choiceinfo_device;
	double *probs;
	CudaSafeCall(cudaMalloc((void**)&distances_device, dist_size));
	CudaSafeCall(cudaMalloc((void**)&costs_device, dist_size));
	CudaSafeCall(cudaMalloc((void**)&tours_device, tours_size));
	CudaSafeCall(cudaMalloc((void**)&visited_device, tours_size));
	CudaSafeCall(cudaMalloc((void**)&choiceinfo_device, ph_size));
	CudaSafeCall(cudaMalloc((void**)&probs, ph_size));
	cudaMemcpy(distances_device, distances, dist_size, cudaMemcpyHostToDevice);
	cudaMemcpy(costs_device, costs, dist_size, cudaMemcpyHostToDevice);
	for (int i = 0; i < n_cities; i++)
		for (int j = 0; j < n_cities; j++)
			pheromones[index(n_cities, i, j)] = initial_pheromone;

	for (int iteration = 0; iteration < n_iterations; iteration++)
	{
		for (int i = 0; i < n_ants; i++)
			for (int j = 0; j < n_cities; j++)
				tours[index(n_cities, i, j)] = Infinity;

		for (int i = 0; i < n_ants; i++)
			for (int j = 0; j < n_cities; j++)
				visited[index(n_cities, i, j)] = 0;

		for (int i = 0; i < n_cities; i++)
		{
			for (int j = 0; j < n_cities; j++)
			{
				double edge_pherom = pheromones[index(n_cities, i, j)];
				double edge_weight = distances[index(n_cities, i, j)];
				double cost_weight = costs[index(n_cities, i, j)];
				double prob = 0.0f;
				if (edge_weight != 0.0f)
				{
					prob = pow(edge_pherom, ALFA)*pow(1 / edge_weight, BETA)*pow(1 / cost_weight, GAMA);
				}
				else
				{
					prob = pow(edge_pherom, ALFA)*pow(Infinity, BETA);
				}
				choiceinfo[index(n_cities, i, j)] = prob;
			}
		}
		cudaMemcpy(choiceinfo_device, choiceinfo, ph_size, cudaMemcpyHostToDevice);
		for (int ant = 0; ant < n_ants; ant++)
		{
			int step = 0;
			int init = rand() % n_cities;
			tours[index(n_cities, ant, step)] = init;
			visited[index(n_cities, ant, init)] = 1;
		}
		cudaMemcpy(visited_device, visited, tours_size, cudaMemcpyHostToDevice);
		cudaMemcpy(tours_device, tours, tours_size, cudaMemcpyHostToDevice);
		cuda_path_traverse << < 1,n_cities >> > (tours_device, visited_device, choiceinfo_device, probs, n_cities);
		CudaCheckError();
		cudaMemcpy(tours, tours_device, tours_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(visited, visited_device, tours_size, cudaMemcpyDeviceToHost);
		evaporate(pheromones, n_cities);
		int *best = optimal_solution(tours, distances, costs, n_ants, n_cities);
		pheromone_update(pheromones, distances,costs, best, n_cities);
	}
	cudaFree(distances_device);
	cudaFree(tours_device);
	cudaFree(visited_device);
	cudaFree(choiceinfo_device);
	cudaFree(probs);
	int *best = optimal_solution(tours, distances,costs, n_ants, n_cities);
	return best;
}