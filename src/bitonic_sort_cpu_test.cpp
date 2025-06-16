#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <cmath>

#define VERBOSE
#include "utils.h"
#include "bitonic_sort_cpu.h"


int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "[ERROR] Wrong number of arguments" << std::endl;
		std::cerr << "USAGE: " << argv[0] << " VECTOR_SIZE_EXP" << std::endl;
		exit(1);
	}

	init_random_generator();

	uint64_t vector_size_exponent = atoi(argv[1]);
	uint64_t vector_size = std::pow(2, vector_size_exponent);
	std::cout << "Vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;
	std::cout << "Allocating " << vector_size * sizeof(int) << " bytes" << std::endl;

	int *v = new int[vector_size];
	
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing..." << std::endl;
	for (int i = 0; i < vector_size; i++) {
		v[i] = std::rand();
		// v[i] = vector_size - i;
	}

	// print_vector(v, vector_size);

	std::cout << "Sorting..." << std::endl;
	clock_t start_time = std::clock();
	// std::sort(v, v + vector_size);
	// sorter(v, 0, vector_size);
	bitonic_sort(v, vector_size);
	clock_t end_time = std::clock();
	
	double elapsed_time_s = (double) (end_time - start_time) / CLOCKS_PER_SEC;
	double elapsed_time_ms = elapsed_time_s * 1000;
	std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;

	// print_vector(v, vector_size);

	std::cout << "Checking if array is sorted correctly..." << std::endl;
	// NOTE: this checks only that the resulting array is monotonic, not that the values don't change from the original ones. For example, an array of all zeros would pass this check.
	for (int i = 0; i < vector_size - 1; i++) {
		if (v[i] > v[i+1]) {
			std::cout << "ERROR: array not sorted: ["
				<< i << "]=" << v[i] << ", ["
				<< i+1 << "]=" << v[i+1] << std::endl;
			exit(1);
		}
	}
	std::cout << "Array sorted successfully!" << std::endl;

	delete[] v;

	return 0;
}

