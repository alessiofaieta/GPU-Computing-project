#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <cmath>
#include <fstream>

#include "utils.h"
#include "bitonic_sort_cpu.h"


int main(int argc, char** argv) {
	if (argc != 5) {
		std::cerr << "[ERROR] Wrong number of arguments" << std::endl;
		std::cerr << "USAGE: " << argv[0]
			<< " INITIAL_VECTOR_SIZE_EXP FINAL_VECTOR_SIZE_EXP"
			<< " N_REPETITIONS CSV_FILE"
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
	const int n_allocated_vectors = 2;
	uint64_t total_allocated_bytes = vector_allocated_bytes * n_allocated_vectors;
	std::cout << "Allocating " << n_allocated_vectors << " * " << vector_allocated_bytes << " = " << total_allocated_bytes << " bytes" << std::endl;

	int n_repetitions = atoi(argv[3]);
	std::cout << "Number of repetitions: " << n_repetitions << std::endl;

	char * csv_file_name = argv[4];
	std::cout << "CSV file: " << csv_file_name << std::endl;
	std::ofstream csv_file(csv_file_name);
	if (!csv_file) {
		std::cerr << "[ERROR] Cannot open file '" << csv_file_name << "'" << std::endl;
		exit(1);
	}
	csv_file << "vector_size_exponent,vector_size,repetition,elapsed_time_ms" << std::endl;


	std::cout << "--- EXECUTION ---" << std::endl;

	int *v_in = new int[final_vector_size];
	int *v_out = new int[final_vector_size];
	
	std::cout << "Random values in [0, " << RAND_MAX << "]" << std::endl;
	std::cout << "Initializing..." << std::endl;
	for (int i = 0; i < final_vector_size; i++) {
		v_in[i] = std::rand();
		// v_in[i] = vector_size - i;
	}

	for (uint64_t vector_size_exponent = initial_vector_size_exponent; vector_size_exponent <= final_vector_size_exponent; vector_size_exponent++) {
		uint64_t vector_size = std::pow(2, vector_size_exponent);

		std::cout << "Current vector size exponent: " << vector_size_exponent << " of [" << initial_vector_size_exponent << "," << final_vector_size_exponent << "]" << std::endl;
		std::cout << "Current vector size: 2 ^ " << vector_size_exponent << " = " << vector_size << std::endl;

		for (int repetition = 1; repetition <= n_repetitions; repetition++) {
			std::cout << "  Sorting [repetition " << repetition << "/" << n_repetitions << "]..." << std::endl;

			// Copy input vector into output vector
			for (int i = 0; i < vector_size; i++) {
				v_out[i] = v_in[i];
			}

			clock_t start_time = std::clock();
			// std::sort(v_out, v_in + vector_size);
			// sorter(v_out, 0, vector_size);
			// bitonic_sort(v_out, vector_size);
			bitonic_sort2(v_out, vector_size);
			clock_t end_time = std::clock();
			
			double elapsed_time_s = (double) (end_time - start_time) / CLOCKS_PER_SEC;
			double elapsed_time_ms = elapsed_time_s * 1000;
			std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;

			csv_file << vector_size_exponent << ","
				<< vector_size << ","
				<< repetition << ","
				<< elapsed_time_ms << std::endl;
		}
	}

	csv_file.close();

	delete[] v_in;
	delete[] v_out;

	return 0;
}
