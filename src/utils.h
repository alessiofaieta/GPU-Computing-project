#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <cstdlib>
#include <ctime>

void init_random_generator() {
	// std::srand(std::time(nullptr));
	std::srand(42);
}

void print_vector(int *V, int N) {
	for (int i = 0; i < N; i++) {
		std::cout << V[i] << ", ";
	}
	std::cout << std::endl;
}

#ifdef VERBOSE
#define LOG(args) std::clog args << std::endl
#else
#define LOG(args)
#endif

#endif
