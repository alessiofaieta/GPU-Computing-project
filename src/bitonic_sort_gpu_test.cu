#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

#include "include/helper_cuda.h"

#define VERBOSE
#include "utils.h"
#include "bitonic_sort_gpu.h"


int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "[ERROR] Wrong number of arguments" << std::endl;
		std::cerr << "USAGE: " << argv[0]
			<< " VECTOR_SIZE_EXP THREADS_PER_BLOCK_EXP COMPARATORS_PER_THREAD_EXP"
			<< " ALGORITHM(A/B/C/D) DO_CPU_CHECK(T/F)" << std::endl;
		exit(1);
	}

	init_random_generator();

	std::cout << "--- CONFIGURATION ---" << std::endl;

	uint64_t vector_size_exponent = atoi(argv[1]);
	uint64_t vector_size = std::pow(2, vector_size_exponent);
	std::cout << "Vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;

	uint64_t vector_allocated_bytes = vector_size * sizeof(int);
	const int n_allocated_vectors = 2;
	uint64_t total_allocated_bytes = vector_allocated_bytes * n_allocated_vectors;
	std::cout << "Allocating " << n_allocated_vectors << " * " << vector_allocated_bytes << " = " << total_allocated_bytes << " bytes" << std::endl;

	uint64_t n_threads_per_block_exponent = atoi(argv[2]);
	uint64_t n_threads_per_block = std::pow(2, n_threads_per_block_exponent);
	std::cout << "Threads per block: 2 ^ " << n_threads_per_block_exponent << " = " << n_threads_per_block << std::endl;
	if (n_threads_per_block < 32 || n_threads_per_block > 1024) {
		std::cerr << "[ERROR] The number of threads per block should be in the range [32, 1024] (exponents in [5, 10])" << std::endl;
		exit(1);
	}

	uint64_t n_comparators_per_thread_exponent = atoi(argv[3]);
	uint64_t n_comparators_per_thread = std::pow(2, n_comparators_per_thread_exponent);
	std::cout << "Comparators per thread: 2 ^ " << n_comparators_per_thread_exponent << " = " << n_comparators_per_thread << std::endl;
	if (n_comparators_per_thread < 1 || n_comparators_per_thread > 16) {
		std::cerr << "[ERROR] The number of comparators per thread should be in the range [1, 16] (exponents in [0, 4])" << std::endl;
		exit(1);
	}

	char * algorithm_param = argv[4];
	char algorithm;
	if (algorithm_param[0] >= 'A' && algorithm_param[0] <= 'D' && algorithm_param[1] == '\0') {
		algorithm = algorithm_param[0];
	} else {
		std::cerr << "ERROR: invalid value for parameter ALGORITHM, expected A/B/C/D but '" << algorithm_param << "' was provided" << std::endl;
		exit(1);
	}
	std::cout << "Algorithm: " << algorithm << " -> " << get_algorithm_name(algorithm) << std::endl;
	
	char * do_cpu_check_param = argv[5];
	bool do_cpu_check;
	if (do_cpu_check_param[0] == 'T' && do_cpu_check_param[1] == '\0') {
		do_cpu_check = true;
	} else if (do_cpu_check_param[0] == 'F' && do_cpu_check_param[1] == '\0') {
		do_cpu_check = false;
	} else {
		std::cerr << "ERROR: invalid value for parameter DO_CPU_CHECK, expected T/F but '" << do_cpu_check_param << "' was provided" << std::endl;
		exit(1);
	}
	
	
	std::cout << "--- EXECUTION ---" << std::endl;

	// Allocate vectors in host memory
	int *v_in = new int[vector_size];
	int *v_out = new int[vector_size];

	// Initialize input vector
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing input vector..." << std::endl;
	for (uint64_t i = 0; i < vector_size; i++) {
		v_in[i] = std::rand();
	}

	// print_vector(v_in, vector_size);

	// Allocate vector in device memory
	int *v_device;
	checkCudaErrors(cudaMalloc(&v_device, vector_allocated_bytes));

	// Copy vector from host memory to device memory
	checkCudaErrors(cudaMemcpy(v_device, v_in, vector_allocated_bytes, cudaMemcpyHostToDevice));

	
	// Create events for timing
	cudaEvent_t gpu_start_time, gpu_end_time;
	cudaEventCreate(&gpu_start_time);
	cudaEventCreate(&gpu_end_time);


	uint64_t n_comparators_per_block = n_comparators_per_thread * n_threads_per_block;
	uint64_t n_values_per_block = N_VALUES_PER_COMPARATOR * n_comparators_per_block;
	uint64_t n_blocks_per_grid = vector_size / n_values_per_block;
	// uint64_t n_blocks_per_grid = (vector_size + n_values_per_block - 1) / n_values_per_block; // formula if `vector_size` is not a power of 2
	uint64_t shared_memory_bytes = n_values_per_block * sizeof(int);
	
	std::cout << "n_threads_per_block=" << n_threads_per_block << " n_blocks_per_grid=" << n_blocks_per_grid << std::endl;
	std::cout << "n_comparators_per_thread=" << n_comparators_per_thread << " n_values_per_block=" << n_values_per_block << std::endl;
	std::cout << "shared_memory_bytes=" << shared_memory_bytes << std::endl;
	assert(n_values_per_block <= vector_size); // ASSERT ERROR: the number of values associated to a block is bigger than the vector size
	
	#define MAX_SHARED_MEMORY_BYTES_PER_BLOCK 49152 // for NVIDIA A30
	assert(shared_memory_bytes <= MAX_SHARED_MEMORY_BYTES_PER_BLOCK); // ASSERT ERROR: the amount of shared memory to allocate is bigger than the available shared memory in a block
	

	// Invoke kernel
	std::cout << "Sorting on GPU..." << std::endl;
	checkCudaErrors(cudaEventRecord(gpu_start_time));

	run_algorithm(algorithm, n_comparators_per_thread, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);

	checkCudaErrors(cudaEventRecord(gpu_end_time));
	checkCudaErrors(cudaEventSynchronize(gpu_end_time));

	// Check for kernel launch errors
	cudaError_t err = cudaGetLastError();
	checkCudaErrors(err);
	if (err != cudaSuccess) {
		std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	float gpu_elapsed_time_ms = 0;
	checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start_time, gpu_end_time));
	std::cout << "Elapsed time: " << gpu_elapsed_time_ms << " ms" << std::endl;

	// Copy result from device memory to host memory
	checkCudaErrors(cudaMemcpy(v_out, v_device, vector_allocated_bytes, cudaMemcpyDeviceToHost));

	// Free device memory
	checkCudaErrors(cudaFree(v_device));

	// print_vector(v_out, vector_size);


	if (do_cpu_check) {
		std::cout << "Sorting on CPU..." << std::endl;
		std::chrono::steady_clock::time_point cpu_start_time = std::chrono::steady_clock::now();
		std::sort(v_in, v_in + vector_size);
		std::chrono::steady_clock::time_point cpu_end_time = std::chrono::steady_clock::now();
	
		long int cpu_elapsed_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end_time - cpu_start_time).count();
		float cpu_elapsed_time_ms = cpu_elapsed_time_ns / 1e6;
		std::cout << "Elapsed time: " << cpu_elapsed_time_ms << " ms" << std::endl;
	
	
		std::cout << "Checking if array is sorted correctly..." << std::endl;
		for (uint64_t i = 0; i < vector_size; i++) {
			if (v_in[i] != v_out[i]) {
				std::cerr << "ERROR: array not sorted at position " << i << ": expected " << v_in[i] << ", found " << v_out[i] << std::endl;
				exit(1);
			}
		}
		
		std::cout << "Array sorted successfully!" << std::endl;
	}

	// Free host memory
	delete[] v_in;
	delete[] v_out;
}
