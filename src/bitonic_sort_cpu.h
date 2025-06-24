#include <cstdint>
#include <algorithm>

#include "utils.h"

#define N_VALUES_PER_COMPARATOR 2


void bitonic_sort_count_swaps(int *v, uint64_t vector_size) {
	uint64_t n_swaps = 0;
	uint64_t n_comparators = vector_size / N_VALUES_PER_COMPARATOR;

	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);

			uint64_t n_comparators_per_half_cleaner = n_values_per_half_cleaner / N_VALUES_PER_COMPARATOR;

			for (uint64_t comparator_id = 0; comparator_id < n_comparators; comparator_id++) {
				uint64_t half_cleaner_id = comparator_id / n_comparators_per_half_cleaner;
				uint64_t comparator_id_within_half_cleaner = comparator_id % n_comparators_per_half_cleaner;
				// uint64_t comparator_id_within_half_cleaner = comparator_id - half_cleaner_id * n_comparators_per_half_cleaner;
				// uint64_t comparator_id_within_half_cleaner = comparator_id & (n_comparators_per_half_cleaner - 1);
				
				uint64_t half_cleaner_value_offset = half_cleaner_id * n_values_per_half_cleaner;
				uint64_t comparator_input1_index = half_cleaner_value_offset + comparator_id_within_half_cleaner;
				
				uint64_t comparator_input2_index;
				if (n_values_per_merger == n_values_per_half_cleaner) {
					comparator_input2_index = half_cleaner_value_offset + n_values_per_half_cleaner - 1 - comparator_id_within_half_cleaner;
				} else {
					comparator_input2_index = comparator_input1_index + n_comparators_per_half_cleaner;
				}

				// LOG(<< "  " << comparator_id << ") half_cleaner_id=" << half_cleaner_id);
				// LOG(<< "    half_cleaner_value_offset=" << half_cleaner_value_offset);
				// LOG(<< "    comparator_id_within_half_cleaner=" << comparator_id_within_half_cleaner);
				// LOG(<< "    Compare indexes " << comparator_input1_index << " and " << comparator_input2_index);

				if (v[comparator_input1_index] > v[comparator_input2_index]) {
					std::swap(v[comparator_input1_index], v[comparator_input2_index]);
					n_swaps++;
				}
			}
		}
	}

	std::cout << vector_size << ") " << n_swaps << std::endl;
}


void bitonic_sort(int *v, uint64_t vector_size) {
	uint64_t n_comparators = vector_size / N_VALUES_PER_COMPARATOR;

	for (uint64_t n_values_per_merger = 2; n_values_per_merger <= vector_size; n_values_per_merger *= 2) {
		for (uint64_t n_values_per_half_cleaner = n_values_per_merger; n_values_per_half_cleaner >= 2; n_values_per_half_cleaner /= 2) {
			LOG(<< "n_values_per_merger=" << n_values_per_merger << " n_values_per_half_cleaner=" << n_values_per_half_cleaner);

			uint64_t n_comparators_per_half_cleaner = n_values_per_half_cleaner / N_VALUES_PER_COMPARATOR;

			for (uint64_t comparator_id = 0; comparator_id < n_comparators; comparator_id++) {
				uint64_t half_cleaner_id = comparator_id / n_comparators_per_half_cleaner;
				uint64_t comparator_id_within_half_cleaner = comparator_id % n_comparators_per_half_cleaner;
				// uint64_t comparator_id_within_half_cleaner = comparator_id - half_cleaner_id * n_comparators_per_half_cleaner;
				// uint64_t comparator_id_within_half_cleaner = comparator_id & (n_comparators_per_half_cleaner - 1);
				
				uint64_t half_cleaner_value_offset = half_cleaner_id * n_values_per_half_cleaner;
				uint64_t comparator_input1_index = half_cleaner_value_offset + comparator_id_within_half_cleaner;
				
				uint64_t comparator_input2_index;
				if (n_values_per_merger == n_values_per_half_cleaner) {
					comparator_input2_index = half_cleaner_value_offset + n_values_per_half_cleaner - 1 - comparator_id_within_half_cleaner;
				} else {
					comparator_input2_index = comparator_input1_index + n_comparators_per_half_cleaner;
				}

				// LOG(<< "  " << comparator_id << ") half_cleaner_id=" << half_cleaner_id);
				// LOG(<< "    half_cleaner_value_offset=" << half_cleaner_value_offset);
				// LOG(<< "    comparator_id_within_half_cleaner=" << comparator_id_within_half_cleaner);
				// LOG(<< "    Compare indexes " << comparator_input1_index << " and " << comparator_input2_index);

				if (v[comparator_input1_index] > v[comparator_input2_index]) {
					std::swap(v[comparator_input1_index], v[comparator_input2_index]);
				}
			}
		}
	}
}

/**
// Recursive implementation
// bitonic sequence -> |bitonic sorter| -> sorted sequence
// two sorted sequences -> |merging network| -> sorted sequence
// bitonic sequence -> |half cleaner| -> two bitonic sequences

void log(char const* phase_name, uint64_t offset, uint64_t n) {
	for (int i = 0; i < n; i++) {
		std::clog << " ";
	}
	std::clog << "[" << phase_name << "] " << offset << "-" << offset + n << std::endl;
}

void half_cleaner(int *v, uint64_t offset, uint64_t len) {
	log("HALF CLEANER", offset, len);
	uint64_t half_len = len / 2;
	for (int i = 0; i < half_len; i++) {
		std::clog << "Compare " << offset + i << " and " << offset + i + half_len << std::endl;
		if (v[offset + i] > v[offset + i + half_len]) {
			std::swap(v[offset + i], v[offset + i + half_len]);
		}
	}
}

void modified_half_cleaner(int *v, uint64_t offset, uint64_t len) {
	log("MODIFIED HALF CLEANER", offset, len);
	uint64_t half_len = len / 2;
	for (int i = 0; i < half_len; i++) {
		std::clog << "Compare " << offset + i << " and " << offset + len - 1 - i << std::endl;
		if (v[offset + i] > v[offset + len - 1 - i]) {
			std::swap(v[offset + i], v[offset + len - 1 - i]);
		}
	}
}

void bitonic_sorter(int *v, uint64_t offset, uint64_t len) {
	log("BITONIC SORTER", offset, len);
	if (len > 1) {
		half_cleaner(v, offset, len);
		uint64_t half_len = len / 2;
		bitonic_sorter(v, offset, half_len);
		bitonic_sorter(v, offset + half_len, half_len);
	}
}

void merger(int *v, uint64_t offset, uint64_t len) {
	log("MERGER", offset, len);
	if (len > 1) {
		modified_half_cleaner(v, offset, len);
		uint64_t half_len = len / 2;
		bitonic_sorter(v, offset, half_len);
		bitonic_sorter(v, offset + half_len, half_len);
	}
}

void sorter(int *v, uint64_t offset, uint64_t len) {
	log("SORTER", offset, len);
	if (len > 1) {
		uint64_t half_len = len / 2;
		sorter(v, offset, half_len);
		sorter(v, offset + half_len, half_len);
		merger(v, offset, len);
	}
}
/**/

