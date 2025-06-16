#include <iostream>
#include <cmath>
#include <algorithm>

#include <cub/cub.cuh>

#include "include/helper_cuda.h"
#include "utils.h"


struct CompareOp {
	__device__ __host__ bool operator()(const int &a, const int &b) const {
		return a < b;
	}
};


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
	const int n_allocated_vectors = 2;
	size_t total_allocated_bytes = vector_allocated_bytes * n_allocated_vectors;
	std::cout << "Allocating " << n_allocated_vectors << " * " << vector_allocated_bytes << " = " << total_allocated_bytes << " bytes" << std::endl;

	// Allocate vectors in host memory
	int *v_in = new int[vector_size];
	int *v_out = new int[vector_size];

	// Initialize input vector
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing input vector..." << std::endl;
	for (int i = 0; i < vector_size; i++) {
		v_in[i] = std::rand();
	}

	// Allocate vector in device memory
	int *v_device;
	checkCudaErrors(cudaMalloc(&v_device, vector_allocated_bytes));

	// Copy vector from host memory to device memory
	checkCudaErrors(cudaMemcpy(v_device, v_in, vector_allocated_bytes, cudaMemcpyHostToDevice));

	cudaEvent_t gpu_start_time, gpu_end_time;
	cudaEventCreate(&gpu_start_time);
	cudaEventCreate(&gpu_end_time);

	// Invoke kernel
	std::cout << "Sorting on GPU..." << std::endl;

	// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceMergeSort.html

	// Determine temporary device storage requirements
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceMergeSort::SortKeys(
		d_temp_storage, temp_storage_bytes,
		v_device, vector_size, CompareOp());

	// Allocate temporary storage
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	// Run sorting operation
	checkCudaErrors(cudaEventRecord(gpu_start_time));
	cub::DeviceMergeSort::SortKeys(
		d_temp_storage, temp_storage_bytes,
		v_device, vector_size, CompareOp());
	checkCudaErrors(cudaEventRecord(gpu_end_time));
	checkCudaErrors(cudaEventSynchronize(gpu_end_time));

	float gpu_elapsed_time_ms = 0;
	checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start_time, gpu_end_time));
	std::cout << "Elapsed time: " << gpu_elapsed_time_ms << " ms" << std::endl;

	// Copy result from device memory to host memory
	checkCudaErrors(cudaMemcpy(v_out, v_device, vector_allocated_bytes, cudaMemcpyDeviceToHost));

	// Free device memory
	checkCudaErrors(cudaFree(v_device));
	checkCudaErrors(cudaFree(d_temp_storage));

	std::cout << "Checking if array is sorted correctly..." << std::endl;
	if (std::is_sorted(v_out, v_out + vector_size)) {
		std::cout << "Array sorted successfully!" << std::endl;
	} else {
		std::cout << "ERROR: array not sorted correctly!" << std::endl;
	}

	// Free host memory
	delete[] v_in;
	delete[] v_out;
}
