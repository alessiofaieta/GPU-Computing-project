
#include "utils.h"


__device__ void sort_comparator_warp(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, uint64_t comparator_id) {
	uint64_t n_comparators_per_half_cleaner = n_values_per_half_cleaner / N_VALUES_PER_COMPARATOR;

	uint64_t half_cleaner_id = comparator_id / n_comparators_per_half_cleaner;
	uint64_t comparator_id_within_half_cleaner = comparator_id % n_comparators_per_half_cleaner;
	
	uint64_t half_cleaner_value_offset = half_cleaner_id * n_values_per_half_cleaner;
	uint64_t comparator_input1_index = half_cleaner_value_offset + comparator_id_within_half_cleaner;

	uint64_t comparator_input2_index;
	if (n_values_per_merger == n_values_per_half_cleaner) { // TODO: remove this check to be more efficient?
		comparator_input2_index = half_cleaner_value_offset + n_values_per_half_cleaner - 1 - comparator_id_within_half_cleaner;
	} else {
		comparator_input2_index = comparator_input1_index + n_comparators_per_half_cleaner;
	}

	bool thread_id_within_comparator = threadIdx.x % 2;
	uint64_t my_index;
	if (thread_id_within_comparator == 0) {
		my_index = comparator_input1_index;
	} else {
		my_index = comparator_input2_index;
	}

	bool other_thread_id_within_comparator = 1 - thread_id_within_comparator;
	int my_value = V[my_index];
	int other_value = __shfl_sync(0xffffffff, my_value, other_thread_id_within_comparator, 2);

	int min_value = min(my_value, other_value);
	int max_value = max(my_value, other_value);

	if (thread_id_within_comparator == 0) {
		V[my_index] = min_value;
	} else {
		V[my_index] = max_value;
	}
}


template <uint64_t N_COMPARATORS_PER_THREAD, uint64_t N_VALUES_PER_THREAD = N_VALUES_PER_COMPARATOR * N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_steps_shared_memory_warp(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_block) {
	extern __shared__ int v_shared[];

	uint64_t block_offset = n_values_per_block * blockIdx.x;

	block_memcpy<N_VALUES_PER_THREAD>(v_shared, V + block_offset);
	
	__syncthreads();

	for (uint64_t n_values_per_half_cleaner = n_values_per_block; n_values_per_half_cleaner > WARP_SIZE; n_values_per_half_cleaner /= 2) {
		#pragma unroll
		for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
			uint64_t comparator_id = blockDim.x * thread_comparator_id + threadIdx.x;
			sort_comparator(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
		}

		__syncthreads();
	}

	#pragma unroll
	for (uint64_t n_values_per_half_cleaner = WARP_SIZE; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
		#pragma unroll
		for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
			uint64_t comparator_id = (blockDim.x * thread_value_id + threadIdx.x) / 2; // associate consecutive pairs of threads to the same comparator
			sort_comparator_warp(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
		}

		__syncwarp();
	}
	
	__syncthreads();

	block_memcpy<N_VALUES_PER_THREAD>(V + block_offset, v_shared);
}

template <uint64_t N_COMPARATORS_PER_THREAD, uint64_t N_VALUES_PER_THREAD = N_VALUES_PER_COMPARATOR * N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_stages_shared_memory_warp(int *V, uint64_t N, uint64_t n_values_per_block) {
	extern __shared__ int v_shared[];

	uint64_t block_offset = n_values_per_block * blockIdx.x;

	block_memcpy<N_VALUES_PER_THREAD>(v_shared, V + block_offset);
	
	__syncthreads();

	#pragma unroll
	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= WARP_SIZE; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			#pragma unroll
			for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
				uint64_t comparator_id = (blockDim.x * thread_value_id + threadIdx.x) / 2; // associate consecutive pairs of threads to the same comparator
				sort_comparator_warp(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
			}

			__syncwarp();
		}
	}

	__syncthreads();


	for (uint64_t n_values_per_merger = WARP_SIZE * 2; n_values_per_merger <= n_values_per_block; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= WARP_SIZE; n_values_per_half_cleaner /= 2) {
			#pragma unroll
			for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
				uint64_t comparator_id = blockDim.x * thread_comparator_id + threadIdx.x;
				sort_comparator(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
			}

			__syncthreads();
		}

		#pragma unroll
		for (uint64_t n_values_per_half_cleaner = WARP_SIZE; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			#pragma unroll
			for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
				uint64_t comparator_id = (blockDim.x * thread_value_id + threadIdx.x) / 2; // associate consecutive pairs of threads to the same comparator
				sort_comparator_warp(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
			}

			__syncwarp();
		}
		
		__syncthreads();
	}

	block_memcpy<N_VALUES_PER_THREAD>(V + block_offset, v_shared);
}



template <uint64_t N_COMPARATORS_PER_THREAD>
void bitonic_sort_shared_memory_grouped_steps_warp(int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t shared_memory_bytes, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	LOG(<< "n_values_per_merger in [2, " << n_values_per_block << "]");
	sort_multiple_stages_shared_memory_warp<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block, shared_memory_bytes>>>(v_device, vector_size, n_values_per_block);

	for (uint64_t n_values_per_merger = n_values_per_block * 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner > n_values_per_block; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);
			sort_one_step<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
		}

		LOG(<< "n_values_per_merger=" << n_values_per_merger << ", n_values_per_half_cleaner in [" << n_values_per_block << ", 2]");
		sort_multiple_steps_shared_memory_warp<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block, shared_memory_bytes>>>(v_device, vector_size, n_values_per_merger, n_values_per_block);
	}
}


