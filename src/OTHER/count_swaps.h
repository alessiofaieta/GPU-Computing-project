
#include "../include/helper_cuda.h"
#include "../utils.h"

#define N_VALUES_PER_COMPARATOR 2

__device__ bool count_swaps_comparator(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, uint64_t comparator_id) {
	uint64_t n_comparators_per_half_cleaner = n_values_per_half_cleaner / N_VALUES_PER_COMPARATOR;

	uint64_t half_cleaner_id = comparator_id / n_comparators_per_half_cleaner;
	uint64_t comparator_id_within_half_cleaner = comparator_id % n_comparators_per_half_cleaner;
	
	uint64_t half_cleaner_value_offset = half_cleaner_id * n_values_per_half_cleaner;
	uint64_t comparator_input1_index = half_cleaner_value_offset + comparator_id_within_half_cleaner;

	uint64_t comparator_input2_index;
	if (n_values_per_merger == n_values_per_half_cleaner) {
		comparator_input2_index = half_cleaner_value_offset + n_values_per_half_cleaner - 1 - comparator_id_within_half_cleaner;
	} else {
		comparator_input2_index = comparator_input1_index + n_comparators_per_half_cleaner;
	}

	bool did_swap = false;

	int comparator_input1 = V[comparator_input1_index];
	int comparator_input2 = V[comparator_input2_index];
	if (comparator_input1 > comparator_input2) {
		V[comparator_input1_index] = comparator_input2;
		V[comparator_input2_index] = comparator_input1;
		did_swap = true;
	}

	return did_swap;
}

__global__ void count_swaps_one_step(uint64_t *v_swap_counters, int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, uint64_t n_comparators_per_block) {
	uint64_t comparator_id = n_comparators_per_block * blockIdx.x + threadIdx.x;
	bool did_swap = count_swaps_comparator(V, N, n_values_per_merger, n_values_per_half_cleaner, comparator_id);

	if (did_swap) {
		v_swap_counters[comparator_id] += 1;
	}
}

uint64_t count_swaps_step_by_step(int *v_device, uint64_t vector_size, uint64_t *swap_counters_device, uint64_t n_comparators) {
	uint64_t n_threads_per_block = 1024;
	if (n_threads_per_block > n_comparators) {
		n_threads_per_block = n_comparators;
	}
	uint64_t n_blocks_per_grid = n_comparators / n_threads_per_block;
	uint64_t n_comparators_per_block = n_threads_per_block;

	uint64_t n_steps = 0;
	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);
			n_steps++;

			count_swaps_one_step<<<n_blocks_per_grid, n_threads_per_block>>>(swap_counters_device, v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
		}
	}

	return n_steps;
}

void count_swaps(uint64_t initial_vector_size_exponent, uint64_t final_vector_size_exponent, uint64_t final_vector_size, int *v_host, int *v_device, uint64_t *swap_counters, uint64_t *steps_counters) {
	uint64_t final_n_comparators = final_vector_size / 2;

	// Allocate vector in host memory
	uint64_t *swap_counters_host = new uint64_t[final_n_comparators];

	// Allocate vector in device memory
	uint64_t *swap_counters_device;
	checkCudaErrors(cudaMalloc(&swap_counters_device, final_n_comparators * sizeof(uint64_t)));

	for (uint64_t vector_size_exponent = initial_vector_size_exponent; vector_size_exponent <= final_vector_size_exponent; vector_size_exponent++) {
		uint64_t vector_size = std::pow(2, vector_size_exponent);
		uint64_t n_comparators = vector_size / N_VALUES_PER_COMPARATOR;

		std::cout << "Current vector size exponent: " << vector_size_exponent << " of [" << initial_vector_size_exponent << "," << final_vector_size_exponent << "]" << std::endl;
		std::cout << "Current vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;
		
		// Initialize vector
		std::fill(swap_counters_host, swap_counters_host + n_comparators, 0);

		// Copy vector from host memory to device memory
		checkCudaErrors(cudaMemcpy(v_device, v_host, vector_size * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(swap_counters_device, swap_counters_host, n_comparators * sizeof(uint64_t), cudaMemcpyHostToDevice));

		// Invoke kernel
		uint64_t n_steps = count_swaps_step_by_step(v_device, vector_size, swap_counters_device, n_comparators);

		cudaDeviceSynchronize(); // Synchronization is required in order to get the last error

		// Check for kernel launch errors
		cudaError_t err = cudaGetLastError();
		checkCudaErrors(err);
		if (err != cudaSuccess) {
			std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
			// exit(1);
		}

		// Copy result from device memory to host memory
		checkCudaErrors(cudaMemcpy(swap_counters_host, swap_counters_device, n_comparators * sizeof(uint64_t), cudaMemcpyDeviceToHost));
		
		uint64_t n_swaps = 0;
		for (int i = 0; i < n_comparators; i++) {
			n_swaps += swap_counters_host[i];
		}

		swap_counters[vector_size_exponent] = n_swaps;
		steps_counters[vector_size_exponent] = n_steps;
	}

	// Free device memory
	checkCudaErrors(cudaFree(swap_counters_device));

	// Free host memory
	delete[] swap_counters_host;
}



	// From `main()` in `bitonic_sort_gpu_measure.cu`
	/**
	std::cout << "Counting number of swaps..." << std::endl;
	uint64_t *swap_counters = new uint64_t[final_vector_size_exponent];
	uint64_t *steps_counters = new uint64_t[final_vector_size_exponent];

	count_swaps(initial_vector_size_exponent, final_vector_size_exponent, final_vector_size, v_host, v_device, swap_counters, steps_counters);

	for (uint64_t vector_size_exponent = initial_vector_size_exponent; vector_size_exponent <= final_vector_size_exponent; vector_size_exponent++) {
		uint64_t vector_size = std::pow(2, vector_size_exponent);
		std::cout << vector_size_exponent << ") vector_size=" << vector_size << " n_swaps=" << swap_counters[vector_size_exponent] << " n_steps=" << steps_counters[vector_size_exponent] << std::endl;
	}


	// TODO: compute effective bandwidth

	for (uint64_t vector_size_exponent = initial_vector_size_exponent; vector_size_exponent <= final_vector_size_exponent; vector_size_exponent++) {
		uint64_t vector_size = std::pow(2, vector_size_exponent);

		uint64_t n_reads;
		uint64_t n_writes;

		switch (algorithm)
		{
		case 'A':
		case 'B':
			n_reads = vector_size * steps_counters[vector_size_exponent];
			n_writes = swap_counters[vector_size_exponent] * N_VALUES_PER_COMPARATOR;
			break;
		case 'C':
			// TODO: more code is required to compute these... is it better to use NCU?
			n_reads = 0;
			n_writes = 0;
			break;
		case 'D':
			n_reads = 0;
			n_writes = 0;
			break;
		}

		uint64_t rw_bytes = (n_reads + n_writes) * sizeof(int);
		std::cout << "Bytes to read/write: " << rw_bytes << " bytes" << std::endl;
	}

	delete[] swap_counters;
	exit(1);

	/**/

