#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>


#include "include/helper_cuda.h"
#include "utils.h"
#include "bitonic_sort_gpu.h"

int main(int argc, char** argv) {
	if (argc != 10) {
		std::cerr << "[ERROR] Wrong number of arguments" << std::endl;
		std::cerr << "USAGE: " << argv[0]
			<< " INITIAL_VECTOR_SIZE_EXP FINAL_VECTOR_SIZE_EXP"
			<< " INITIAL_THREADS_PER_BLOCK_EXP FINAL_THREADS_PER_BLOCK_EXP"
			<< " INITIAL_COMPARATORS_PER_THREAD_EXP FINAL_COMPARATORS_PER_THREAD_EXP"
			<< " N_REPETITIONS ALGORITHM(A/B/C/D) CSV_FILE"
			<< std::endl;
		exit(1);
	}

	init_random_generator();

	std::cout << "--- CONFIGURATION ---" << std::endl;

	uint64_t initial_vector_size_exponent = atoi(argv[1]);
	uint64_t final_vector_size_exponent = atoi(argv[2]);
	uint64_t initial_vector_size = std::pow(2, initial_vector_size_exponent);
	uint64_t final_vector_size = std::pow(2, final_vector_size_exponent);
	std::cout << "Initial vector size: 2 ^ " << initial_vector_size_exponent << " = " << initial_vector_size << std::endl;
	std::cout << "Final vector size: 2 ^ " << final_vector_size_exponent << " = " << final_vector_size << std::endl;

	uint64_t vector_allocated_bytes = final_vector_size * sizeof(int);
	std::cout << "Allocating " << vector_allocated_bytes << " bytes" << std::endl;

	uint64_t initial_threads_per_block_exponent = atoi(argv[3]);
	uint64_t final_threads_per_block_exponent = atoi(argv[4]);
	uint64_t initial_threads_per_block = std::pow(2, initial_threads_per_block_exponent);
	uint64_t final_threads_per_block = std::pow(2, final_threads_per_block_exponent);
	std::cout << "Initial threads per block: 2 ^ " << initial_threads_per_block_exponent << " = " << initial_threads_per_block << std::endl;
	std::cout << "Final threads per block: 2 ^ " << final_threads_per_block_exponent << " = " << final_threads_per_block << std::endl;
	if (initial_threads_per_block < 32 || initial_threads_per_block > 1024 || final_threads_per_block < 32 || final_threads_per_block > 1024) {
		std::cerr << "[ERROR] The number of threads per block should be in the range [32, 1024] (exponents in [5, 10])" << std::endl;
		exit(1);
	}

	uint64_t initial_comparators_per_thread_exponent = atoi(argv[5]);
	uint64_t final_comparators_per_thread_exponent = atoi(argv[6]);
	uint64_t initial_comparators_per_thread = std::pow(2, initial_comparators_per_thread_exponent);
	uint64_t final_comparators_per_thread = std::pow(2, final_comparators_per_thread_exponent);
	std::cout << "Initial comparators per thread: 2 ^ " << initial_comparators_per_thread_exponent << " = " << initial_comparators_per_thread << std::endl;
	std::cout << "Final comparators per thread: 2 ^ " << final_comparators_per_thread_exponent << " = " << final_comparators_per_thread << std::endl;
	if (initial_comparators_per_thread < 1 || initial_comparators_per_thread > 16 || final_comparators_per_thread < 1 || final_comparators_per_thread > 16) {
		std::cerr << "[ERROR] The number of comparators per thread should be in the range [1, 16] (exponents in [0, 4])" << std::endl;
		exit(1);
	}

	int n_repetitions = atoi(argv[7]);
	std::cout << "Number of repetitions: " << n_repetitions << std::endl;
	
	char * algorithm_param = argv[8];
	char algorithm;
	if (algorithm_param[0] >= 'A' && algorithm_param[0] <= 'D' && algorithm_param[1] == '\0') {
		algorithm = algorithm_param[0];
	} else {
		std::cerr << "ERROR: invalid value for parameter ALGORITHM, expected A/B/C/D but '" << algorithm_param << "' was provided" << std::endl;
		exit(1);
	}
	std::cout << "Algorithm: " << algorithm << " -> " << get_algorithm_name(algorithm) << std::endl;

	char * csv_file_name = argv[9];
	std::cout << "CSV file: " << csv_file_name << std::endl;
	std::ofstream csv_file(csv_file_name);
	if (!csv_file) {
		std::cerr << "[ERROR] Cannot open file '" << csv_file_name << "'" << std::endl;
		exit(1);
	}
	csv_file << "vector_size_exponent,vector_size,threads_per_block_exponent,threads_per_block,comparators_per_thread_exponent,comparators_per_thread,values_per_block,blocks_per_grid,repetition,elapsed_time_ms" << std::endl;


	std::cout << "--- EXECUTION ---" << std::endl;

	// Allocate vector in host memory
	int *v_host = new int[final_vector_size];

	// Initialize vector
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing vector..." << std::endl;
	for (uint64_t i = 0; i < final_vector_size; i++) {
		v_host[i] = std::rand();
	}

	// Allocate vector in device memory
	int *v_device;
	checkCudaErrors(cudaMalloc(&v_device, vector_allocated_bytes));

	// Warm up the GPU (otherwise the first execution takes longer than the following ones)
	// sort_one_step<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
	sort_one_step<1><<<1, 256>>>(v_device, final_vector_size, 2, 2, 256);


	// Create events for timing
	cudaEvent_t gpu_start_time, gpu_end_time;
	cudaEventCreate(&gpu_start_time);
	cudaEventCreate(&gpu_end_time);

	for (uint64_t vector_size_exponent = initial_vector_size_exponent; vector_size_exponent <= final_vector_size_exponent; vector_size_exponent++) {
		uint64_t vector_size = std::pow(2, vector_size_exponent);
		
		std::cout << "Current vector size exponent: " << vector_size_exponent << " of [" << initial_vector_size_exponent << "," << final_vector_size_exponent << "]" << std::endl;
		std::cout << "Current vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;

		for (uint64_t n_threads_per_block_exponent = initial_threads_per_block_exponent; n_threads_per_block_exponent <= final_threads_per_block_exponent; n_threads_per_block_exponent++) {
			uint64_t n_threads_per_block = std::pow(2, n_threads_per_block_exponent);
			
			std::cout << "  Current threads per block exponent: " << n_threads_per_block_exponent << " of [" << initial_threads_per_block_exponent << "," << final_threads_per_block_exponent << "]" << std::endl;
			std::cout << "  Current threads per block: 2 ^ " << n_threads_per_block_exponent << " = " << n_threads_per_block << std::endl;

			for (uint64_t n_comparators_per_thread_exponent = initial_comparators_per_thread_exponent; n_comparators_per_thread_exponent <= final_comparators_per_thread_exponent; n_comparators_per_thread_exponent++) {
				uint64_t n_comparators_per_thread = std::pow(2, n_comparators_per_thread_exponent);

				uint64_t n_comparators_per_block = n_comparators_per_thread * n_threads_per_block;
				uint64_t n_values_per_block = N_VALUES_PER_COMPARATOR * n_comparators_per_block;
				uint64_t n_blocks_per_grid = vector_size / n_values_per_block;
				uint64_t shared_memory_bytes = n_values_per_block * sizeof(int);
				
				std::cout << "    Current comparators per thread exponent: " << n_comparators_per_thread_exponent << " of [" << initial_comparators_per_thread_exponent << "," << final_comparators_per_thread_exponent << "]" << std::endl;
				std::cout << "    Current comparators per thread: 2 ^ " << n_comparators_per_thread_exponent << " = " << n_comparators_per_thread << std::endl;
				std::cout << "    Current values per block: " << n_values_per_block << std::endl;
				std::cout << "    Current blocks per grid: " << n_blocks_per_grid << std::endl;
				std::cout << "    Current bytes to allocate for shared memory: " << shared_memory_bytes << std::endl;
				
				if (n_values_per_block > vector_size) {
					std::cout << "    Skipping... (n_values_per_block > vector_size)" << std::endl;
					break;
				}
				
				#define MAX_SHARED_MEMORY_BYTES_PER_BLOCK 49152 // for NVIDIA A30
				if (shared_memory_bytes > MAX_SHARED_MEMORY_BYTES_PER_BLOCK) { // TODO: skip only if the algorithm uses shared memory?
					std::cout << "    Skipping... (shared_memory_bytes > MAX_SHARED_MEMORY_BYTES_PER_BLOCK)" << std::endl;
					break;
				}
				
				
				for (int repetition = 1; repetition <= n_repetitions; repetition++) {
					std::cout << "      Sorting [repetition " << repetition << "/" << n_repetitions << "]..." << std::endl;
					
					// Copy vector from host memory to device memory
					uint64_t vector_size_in_bytes = vector_size * sizeof(int);
					checkCudaErrors(cudaMemcpy(v_device, v_host, vector_size_in_bytes, cudaMemcpyHostToDevice));
					
					
					// Invoke kernel
					checkCudaErrors(cudaEventRecord(gpu_start_time));
					
					run_algorithm(algorithm, n_comparators_per_thread, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);

					checkCudaErrors(cudaEventRecord(gpu_end_time));
					checkCudaErrors(cudaEventSynchronize(gpu_end_time));
					
					// Check for kernel launch errors
					cudaError_t err = cudaGetLastError();
					checkCudaErrors(err);
					if (err != cudaSuccess) {
						std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
						// exit(1);
					}

					float gpu_elapsed_time_ms = 0;
					if (err != cudaSuccess) {
						gpu_elapsed_time_ms = NAN;
					} else {
						checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, gpu_start_time, gpu_end_time));
					}
					std::cout << "      Elapsed time: " << gpu_elapsed_time_ms << " ms" << std::endl;

					csv_file << vector_size_exponent << ","
						<< vector_size << ","
						<< n_threads_per_block_exponent << ","
						<< n_threads_per_block << ","
						<< n_comparators_per_thread_exponent << ","
						<< n_comparators_per_thread << ","
						<< n_values_per_block << ","
						<< n_blocks_per_grid << ","
						<< repetition << ","
						<< gpu_elapsed_time_ms << std::endl;
				}
			}
		}
	}

	csv_file.close();

	// Free device memory
	checkCudaErrors(cudaFree(v_device));

	// Free host memory
	delete[] v_host;
}
