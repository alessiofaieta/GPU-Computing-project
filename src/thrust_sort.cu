#include <iostream>
#include <cmath>
#include <algorithm>

#include <thrust/sort.h>

#include "include/helper_cuda.h"
#include "utils.h"


int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "[ERROR] Wrong number of arguments" << std::endl;
		std::cerr << "USAGE: " << argv[0] << " VECTOR_SIZE_EXP" << std::endl;
		exit(1);
	}
	
	init_random_generator();

	size_t vector_size_exponent = atoi(argv[1]);
	size_t vector_size = std::pow(2, vector_size_exponent);
	std::cout << "Vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;
	size_t vector_allocated_bytes = vector_size * sizeof(int);
	std::cout << "Allocating " << vector_allocated_bytes << " bytes" << std::endl;

	// Allocate vectors in host memory
	int *v_in = new int[vector_size];

	// Initialize input vector
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing input vector..." << std::endl;
	for (int i = 0; i < vector_size; i++) {
		v_in[i] = std::rand();
	}

	cudaEvent_t gpu_start_time, gpu_end_time;
	cudaEventCreate(&gpu_start_time);
	cudaEventCreate(&gpu_end_time);

	// Invoke kernel
	std::cout << "Sorting on GPU..." << std::endl;
	checkCudaErrors(cudaEventRecord(gpu_start_time));

	// https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga01621fff7b6eb24fb68944cc2f10af6a.html
	thrust::sort(v_in, v_in + vector_size);
	
	checkCudaErrors(cudaEventRecord(gpu_end_time));
	checkCudaErrors(cudaEventSynchronize(gpu_end_time));

	float gpu_elapsed_time_ms = 0;
	checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start_time, gpu_end_time));
	std::cout << "Elapsed time: " << gpu_elapsed_time_ms << " ms" << std::endl;

	std::cout << "Checking if array is sorted correctly..." << std::endl;
	if (std::is_sorted(v_in, v_in + vector_size)) {
		std::cout << "Array sorted successfully!" << std::endl;
	} else {
		std::cout << "ERROR: array not sorted correctly!" << std::endl;
	}

	// Free host memory
	delete[] v_in;
}
