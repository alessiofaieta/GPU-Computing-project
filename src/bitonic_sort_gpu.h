
#include "utils.h"
#include <cstdint>

#define WARP_SIZE 32
#define N_VALUES_PER_COMPARATOR 2

// Device code

__device__ void sort_comparator(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, uint64_t comparator_id) {
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

	int comparator_input1 = V[comparator_input1_index];
	int comparator_input2 = V[comparator_input2_index];
	if (comparator_input1 > comparator_input2) {
		V[comparator_input1_index] = comparator_input2;
		V[comparator_input2_index] = comparator_input1;
	}
	
	/**
	int min_value = min(V[comparator_input1_index], V[comparator_input2_index]);
	int max_value = max(V[comparator_input1_index], V[comparator_input2_index]);
	V[comparator_input1_index] = min_value;
	V[comparator_input2_index] = max_value;
	/**
	// NOTE: If the threads in the second half of the warp read the second input before of the first input, then there shouldn't be a shared memory conflict with the threads of the first half.
	// bool warp_half_id = threadIdx.x & 16;
	bool warp_half_id = threadIdx.x % 32 < 16;

	int input1_id, input2_id;
	if (warp_half_id) {
		input1_id = comparator_input1_index;
		input2_id = comparator_input2_index;
	} else {
		input1_id = comparator_input2_index;
		input2_id = comparator_input1_index;
	}

	int input1 = V[input1_id];
	int input2 = V[input2_id];

	int comparator_input1, comparator_input2;
	if (warp_half_id) {
		comparator_input1 = input1;
		comparator_input2 = input2;
	} else {
		comparator_input1 = input2;
		comparator_input2 = input1;
	}
	
	input1_id = comparator_input1_index;
	input2_id = comparator_input2_index;

	if (comparator_input1 > comparator_input2) {
		V[input1_id] = comparator_input2;
		V[input2_id] = comparator_input1;
	}
	/**/
}

template <uint64_t N_VALUES_PER_THREAD>
__device__ void block_memcpy(int *dst, int *src) {
	#pragma unroll
	for (uint64_t i = 0; i < N_VALUES_PER_THREAD; i++) {
		dst[blockDim.x * i + threadIdx.x] = src[blockDim.x * i + threadIdx.x];
	}
}


template <uint64_t N_COMPARATORS_PER_THREAD>
__global__ void sort_one_step(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, uint64_t n_comparators_per_block) {
	#pragma unroll
	for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
		uint64_t comparator_id = n_comparators_per_block * blockIdx.x + blockDim.x * thread_comparator_id + threadIdx.x;
		sort_comparator(V, N, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
	}
}

template <uint64_t N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_steps(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	for (uint64_t n_values_per_half_cleaner = n_values_per_block; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
		#pragma unroll
		for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
			uint64_t comparator_id = n_comparators_per_block * blockIdx.x + blockDim.x * thread_comparator_id + threadIdx.x;
			sort_comparator(V, N, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
		}

		__syncthreads();
	}
}

template <uint64_t N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_stages(int *V, uint64_t N, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= n_values_per_block; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			#pragma unroll
			for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
				uint64_t comparator_id = n_comparators_per_block * blockIdx.x + blockDim.x * thread_comparator_id + threadIdx.x;
				sort_comparator(V, N, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
			}
	
			__syncthreads();
		}
	}
}


template <uint64_t N_COMPARATORS_PER_THREAD, uint64_t N_VALUES_PER_THREAD = N_VALUES_PER_COMPARATOR * N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_steps_shared_memory(int *V, uint64_t N, uint64_t n_values_per_merger, uint64_t n_values_per_block) {
	extern __shared__ int v_shared[];

	uint64_t block_offset = n_values_per_block * blockIdx.x;

	block_memcpy<N_VALUES_PER_THREAD>(v_shared, V + block_offset);

	__syncthreads();

	for (uint64_t n_values_per_half_cleaner = n_values_per_block; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
		#pragma unroll
		for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
			uint64_t comparator_id = blockDim.x * thread_comparator_id + threadIdx.x;
			sort_comparator(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
		}

		__syncthreads();
	}

	block_memcpy<N_VALUES_PER_THREAD>(V + block_offset, v_shared);
}

template <uint64_t N_COMPARATORS_PER_THREAD, uint64_t N_VALUES_PER_THREAD = N_VALUES_PER_COMPARATOR * N_COMPARATORS_PER_THREAD>
__global__ void sort_multiple_stages_shared_memory(int *V, uint64_t N, uint64_t n_values_per_block) {
	extern __shared__ int v_shared[];

	uint64_t block_offset = n_values_per_block * blockIdx.x;

	block_memcpy<N_VALUES_PER_THREAD>(v_shared, V + block_offset);
	
	__syncthreads();

	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= n_values_per_block; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			#pragma unroll
			for (uint64_t thread_comparator_id = 0; thread_comparator_id < N_COMPARATORS_PER_THREAD; thread_comparator_id++) {
				uint64_t comparator_id = blockDim.x * thread_comparator_id + threadIdx.x;
				sort_comparator(v_shared, n_values_per_block, n_values_per_merger, n_values_per_half_cleaner, comparator_id);
			}

			__syncthreads();
		}
	}

	block_memcpy<N_VALUES_PER_THREAD>(V + block_offset, v_shared);
}


__device__ int sort_comparator_warp(uint64_t n_values_per_merger, uint64_t n_values_per_half_cleaner, int my_value, uint64_t lane_id) {
	uint64_t n_comparators_per_half_cleaner = n_values_per_half_cleaner / N_VALUES_PER_COMPARATOR;

	uint64_t half_cleaner_id = lane_id / n_values_per_half_cleaner;
	uint64_t lane_id_within_half_cleaner = lane_id % n_values_per_half_cleaner;

	uint64_t half_cleaner_value_offset = half_cleaner_id * n_values_per_half_cleaner;

	uint64_t other_lane_id_within_half_cleaner;
	if (n_values_per_merger == n_values_per_half_cleaner) {
		other_lane_id_within_half_cleaner = n_values_per_half_cleaner - 1 - lane_id_within_half_cleaner;
	} else {
		other_lane_id_within_half_cleaner = (lane_id_within_half_cleaner + n_comparators_per_half_cleaner) % n_values_per_half_cleaner;
	}

	uint64_t other_lane_id = half_cleaner_value_offset + other_lane_id_within_half_cleaner;


	int other_value = __shfl_sync(0xffffffff, my_value, other_lane_id, 32);

	int min_value = min(my_value, other_value);
	int max_value = max(my_value, other_value);

	if (lane_id < other_lane_id) {
		my_value = min_value;
	} else {
		my_value = max_value;
	}

	return my_value;
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
	for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
		uint64_t thread_id = blockDim.x * thread_value_id + threadIdx.x;
		int my_value = v_shared[thread_id];
		int laneId = threadIdx.x & 0x1f;

		#pragma unroll
		for (uint64_t n_values_per_half_cleaner = WARP_SIZE; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			my_value = sort_comparator_warp(n_values_per_merger, n_values_per_half_cleaner, my_value, laneId);
		}

		v_shared[thread_id] = my_value;
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
	for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
		uint64_t thread_id = blockDim.x * thread_value_id + threadIdx.x;
		int my_value = v_shared[thread_id];
		int laneId = threadIdx.x & 0x1f;

		#pragma unroll
		for (uint64_t n_values_per_merger = 2; n_values_per_merger <= WARP_SIZE; n_values_per_merger *= 2) {
			for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
				my_value = sort_comparator_warp(n_values_per_merger, n_values_per_half_cleaner, my_value, laneId);
			}

			v_shared[thread_id] = my_value;
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
		for (uint64_t thread_value_id = 0; thread_value_id < N_VALUES_PER_THREAD; thread_value_id++) {
			uint64_t thread_id = blockDim.x * thread_value_id + threadIdx.x;
			int my_value = v_shared[thread_id];
			int laneId = threadIdx.x & 0x1f;

			#pragma unroll
			for (uint64_t n_values_per_half_cleaner = WARP_SIZE; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
				my_value = sort_comparator_warp(n_values_per_merger, n_values_per_half_cleaner, my_value, laneId);
			}

			v_shared[thread_id] = my_value;
		}
		
		__syncthreads();
	}

	block_memcpy<N_VALUES_PER_THREAD>(V + block_offset, v_shared);
}



// Host code

template <uint64_t N_COMPARATORS_PER_THREAD>
void bitonic_sort_global_memory_step_by_step(int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t n_comparators_per_block) {
	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);

			sort_one_step<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
		}
	}
}

template <uint64_t N_COMPARATORS_PER_THREAD>
void bitonic_sort_global_memory_grouped_steps(int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	LOG(<< "n_values_per_merger in [2, " << n_values_per_block << "]");
	sort_multiple_stages<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_block, n_comparators_per_block);

	for (uint64_t n_values_per_merger = n_values_per_block * 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner > n_values_per_block; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);
			sort_one_step<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
		}

		LOG(<< "n_values_per_merger=" << n_values_per_merger << ", n_values_per_half_cleaner in [" << n_values_per_block << ", 2]");
		sort_multiple_steps<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_block, n_comparators_per_block);
	}
}

template <uint64_t N_COMPARATORS_PER_THREAD>
void bitonic_sort_shared_memory_grouped_steps(int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t shared_memory_bytes, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	LOG(<< "n_values_per_merger in [2, " << n_values_per_block << "]");
	sort_multiple_stages_shared_memory<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block, shared_memory_bytes>>>(v_device, vector_size, n_values_per_block);

	for (uint64_t n_values_per_merger = n_values_per_block * 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner > n_values_per_block; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);
			sort_one_step<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block>>>(v_device, vector_size, n_values_per_merger, n_values_per_half_cleaner, n_comparators_per_block);
		}

		LOG(<< "n_values_per_merger=" << n_values_per_merger << ", n_values_per_half_cleaner in [" << n_values_per_block << ", 2]");
		sort_multiple_steps_shared_memory<N_COMPARATORS_PER_THREAD><<<n_blocks_per_grid, n_threads_per_block, shared_memory_bytes>>>(v_device, vector_size, n_values_per_merger, n_values_per_block);
	}
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

const char* get_algorithm_name(char algorithm) {
	const char* algorithm_name;
	switch (algorithm)
	{
	case 'A':
		algorithm_name = "bitonic_sort_global_memory_step_by_step";
		break;
	case 'B':
		algorithm_name = "bitonic_sort_global_memory_grouped_steps";
		break;
	case 'C':
		algorithm_name = "bitonic_sort_shared_memory_grouped_steps";
		break;
	case 'D':
		algorithm_name = "bitonic_sort_shared_memory_grouped_steps_warp";
		break;
	}

	return algorithm_name;
}

template <uint64_t N_COMPARATORS_PER_THREAD>
void run_algorithm_template(char algorithm, int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t shared_memory_bytes, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	switch (algorithm)
	{
	case 'A':
		bitonic_sort_global_memory_step_by_step<N_COMPARATORS_PER_THREAD>(v_device, vector_size, n_blocks_per_grid, n_threads_per_block, n_comparators_per_block);
		break;
	case 'B':
		bitonic_sort_global_memory_grouped_steps<N_COMPARATORS_PER_THREAD>(v_device, vector_size, n_blocks_per_grid, n_threads_per_block, n_values_per_block, n_comparators_per_block);
		break;
	case 'C':
		bitonic_sort_shared_memory_grouped_steps<N_COMPARATORS_PER_THREAD>(v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	case 'D':
		bitonic_sort_shared_memory_grouped_steps_warp<N_COMPARATORS_PER_THREAD>(v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	}
}

void run_algorithm(char algorithm, uint64_t n_comparators_per_thread, int *v_device, uint64_t vector_size, uint64_t n_blocks_per_grid, uint64_t n_threads_per_block, uint64_t shared_memory_bytes, uint64_t n_values_per_block, uint64_t n_comparators_per_block) {
	switch (n_comparators_per_thread)
	{
	case 1:
		run_algorithm_template<1>(algorithm, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	case 2:
		run_algorithm_template<2>(algorithm, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	case 4:
		run_algorithm_template<4>(algorithm, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	case 8:
		run_algorithm_template<8>(algorithm, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	case 16:
		run_algorithm_template<16>(algorithm, v_device, vector_size, n_blocks_per_grid, n_threads_per_block, shared_memory_bytes, n_values_per_block, n_comparators_per_block);
		break;
	}
}



