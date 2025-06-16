
# GPU Computing - Project

This repository contains the code for the project of the "GPU Computing" course at the University of Trento, A.A. 2023/2024.

The goal of this project is to implement a parallel sorting algorithm over GPU using CUDA. The selected algorithm is *bitonic sort*.

The code in this repository allows to execute the algorithm implementations, check their correctness, measure their performance and produce plots of their performance.

The information about the performance of the algorithm implementations is stored in CSV files.



## Reproducibility TL;DR

In order to produce all the CSV files on the university cluster, follow these steps.

Setup:
- create the directory to contain the CSV files:
  ```sh
  mkdir -p csv
  ```
- load the CUDA module:
  ```sh
  module load CUDA/12.3.2
  ```

Compile all executables:
- open an interactive Slurm shell session without allocating any GPU:
  ```sh
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --partition=cpu --pty bash
  ```

  _Note_: if the `cpu` partition is not available, use the following command:
  ```sh
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 --partition=edu-short --pty bash
  ```
- from inside the interactive shell session, compile the executables:
  ```sh
  make all
  ```
- close the interactive shell session:
  ```sh
  exit
  ```

Produce the CSV files:
- execute the CPU implementations:
  ```sh
  sbatch script/run_cpu_execs.sh
  ```
- execute the GPU implementations:
  ```sh
  sbatch script/run_gpu_execs.sh
  ```
- execute the GPU libraries implementations:
  ```sh
  sbatch script/run_lib_execs.sh
  ```



## Project structure

The `src` directory contains the source code for the project. There are three types of sources:
1. CPU implementations: these files contain the CPU implementations of the algorithm.
    - `bitonic_sort_cpu_test.cpp` tests the correctness of the algorithm;
    - `bitonic_sort_cpu_measure.cpp` measures the execution time of the algorithm.
2. GPU implementations: these files contain the GPU implementations of the algorithm.
    - `bitonic_sort_gpu_test.cu` tests the correctness of the algorithm;
    - `bitonic_sort_gpu_measure.cu` measures the execution time of the algorithm.
3. GPU libraries implementations: these files use a GPU sorting algorithm implementation from some CUDA library.
    - `thrust_sort.cu` uses `thrust::sort()` from the [Thrust library](https://nvidia.github.io/cccl/thrust/api.html) (documentation available [here](https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1ga01621fff7b6eb24fb68944cc2f10af6a.html)).
    - `cub_merge_sort.cu` uses `cub::DeviceMergeSort::SortKeys()` from the [CUB library](https://nvidia.github.io/cccl/cub/) (documentation available [here](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceMergeSort.html)).
    - `cub_radix_sort.cu` uses `cub::DeviceRadixSort::SortKeys()` from the [CUB library](https://nvidia.github.io/cccl/cub/) (documentation available [here](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html)).

The `script` directory contains some useful shell scripts:
- `bitonic_sort_lib_measure.sh`: script that runs a GPU library implementation multiple times and for multiple vector sizes, and produces a CSV output.
- `run_cpu_execs.sh`: batch script to execute the CPU implementations with `sbatch`.
- `run_gpu_execs.sh`: batch script to execute the GPU implementations with `sbatch`.
- `run_lib_execs.sh`: batch script to execute the GPU libraries implementations with `sbatch`.
- `sbatch_run_command.sh`: batch script to submit a job with `sbatch` allocating one GPU in order to execute a command.
- `sbatch_profile_command.sh`: batch script to submit a job with `sbatch` allocating one GPU in order to profile a command with NVIDIA Nsight Compute CLI (`ncu`).

The `csv_results` directory contains the CSV outputs of the algorithms executions on the university cluster.

The `plots` directory contains the code to produce the plots, and the plots themselves.



## Using the university cluster

To use the **university cluster** with Slurm, these are the options:
- open an interactive shell session without allocating any GPU:
  ```sh
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --partition=cpu --pty bash
  ```
  This option is useful for compiling a CPU or GPU implementation and for executing a CPU implementation.

  _Note_: if the `cpu` partition is not available, use the following command:
  ```sh
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 --partition=edu-short --pty bash
  ```
- open an interactive shell session allocating one GPU:
  ```sh
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:a30.24:1 --partition=edu-short --pty bash
  ```
  This option is useful for executing a GPU implementation.
- submit a job executing the command `[COMMAND]` allocating one GPU:
  ```sh
  sbatch script/sbatch_run_command.sh [COMMAND]
  ```
  This option is useful for executing a GPU implementation.

The available partitions are `edu-short`, with a maximum execution time of 5 minutes, and `edu-medium`, with a maximum execution time of 2 hours. They can be used with both the `srun` and `sbatch` commands. In the case of `sbatch`, it will override the default specified in `script/sbatch_run_command.sh`, which is `edu-short`.

Before compiling or executing a GPU implementation, remember to load the CUDA module by executing `module load CUDA/12.3.2`.



## Compiling the project

To compile the source code, there are specific `make` targets for each type of implementation:
- `make build_cpu_execs` for the CPU implementations;
- `make build_gpu_execs` for the GPU implementations;
- `make build_lib_execs` for the GPU libraries implementations.

There is also the `make all` target, that compiles all the implementations.

Each executable can also be compiled individually by executing `make EXECUTABLE_PATH`, e.g. `make bin/bitonic_sort_gpu_test`.

The executables are stored in the `bin` directory. To delete the `bin` directory, run `make clean`.

The CPU implementations are compiled to multiple binaries, one for each optimization level (from 0 to 3). The name of each executable corresponds to the name of the source file followed by the optimization level, for example `bitonic_sort_cpu_measure0` or `bitonic_sort_cpu_test3`.



## Running the project

Each executable expects some *input parameters*. Running an executable with the wrong number of parameters prints a summary of how to use it.


### bitonic_sort_cpu_test

The `bitonic_sort_cpu_test` executable runs the CPU implementation of the sorting algorithm, measures its execution time and then checks that the array is sorted correctly.

It is compiled with four different optimization levels, from 0 to 3. The corresponding executable names are `bitonic_sort_cpu_test0`, `bitonic_sort_cpu_test1`, `bitonic_sort_cpu_test2` and `bitonic_sort_cpu_test3`.

The **input parameters** of the executable are:
- `VECTOR_SIZE_EXP`: the size of the vector, given as the exponent of the desired power of $2$.

Usage **example**: `bin/bitonic_sort_cpu_test0 20`


### bitonic_sort_cpu_measure

The `bitonic_sort_cpu_measure` executable runs the CPU implementation of the sorting algorithm multiple times according to the input parameters, measures the elapsed time for each execution and writes it in a CSV file.

It is compiled with four different optimization levels, from 0 to 3. The corresponding executable names are `bitonic_sort_cpu_measure0`, `bitonic_sort_cpu_measure1`, `bitonic_sort_cpu_measure2` and `bitonic_sort_cpu_measure3`.

The **input parameters** of the executable are:
- `INITIAL_VECTOR_SIZE_EXP` and `FINAL_VECTOR_SIZE_EXP`: the first and last size of the vector to loop through, given as the exponents of the desired powers of $2$.
- `N_REPETITIONS`: the number of times to repeat the execution of the algorithm.
- `CSV_FILE`: the path to the output CSV file.

The fields stored in the **CSV output file** are:
- `vector_size_exponent`: the size of the vector, given as the exponent of the corresponding power of $2$.
- `vector_size`: the size of the vector.
- `repetition`: the index of the current repetition of the execution of the algorithm.
- `elapsed_time_ms`: the execution time of the algorithm in milliseconds.

In order to ignore the CSV output, pass `/dev/null` as value for the `CSV_FILE` parameter.

Usage **examples** (all optimization levels):
- `bin/bitonic_sort_cpu_measure0 11 23 10 csv/bitonic_sort_cpu_measure0.csv`
- `bin/bitonic_sort_cpu_measure1 11 23 10 csv/bitonic_sort_cpu_measure1.csv`
- `bin/bitonic_sort_cpu_measure2 11 23 10 csv/bitonic_sort_cpu_measure2.csv`
- `bin/bitonic_sort_cpu_measure3 11 23 10 csv/bitonic_sort_cpu_measure3.csv`


### bitonic_sort_gpu_test

The `bitonic_sort_gpu_test` executable runs the GPU implementation of the sorting algorithm, measures its execution time and then checks that the array is sorted correctly.

The **input parameters** of the executable are:
- `VECTOR_SIZE_EXP`: the size of the vector, given as the exponent of the desired power of $2$.
- `THREADS_PER_BLOCK_EXP`: the number of threads per block, given as the exponent of the desired power of $2$.
- `COMPARATORS_PER_THREAD_EXP`: the number of comparators per thread, given as the exponent of the desired power of $2$.
- `ALGORITHM`: the algorithm to execute. Valid options are:
  - `A` for `bitonic_sort_global_memory_step_by_step`
  - `B` for `bitonic_sort_global_memory_grouped_steps`
  - `C` for `bitonic_sort_shared_memory_grouped_steps`
  - `D` for `bitonic_sort_shared_memory_grouped_steps_warp`
- `DO_CPU_CHECK`: whether to check if the sort is correct; valid options are `T` for `true` and `F` for `false`.

Usage **example**: `bin/bitonic_sort_gpu_test 25 8 0 A T`


### bitonic_sort_gpu_measure

The `bitonic_sort_gpu_measure` executable runs the GPU implementation of the sorting algorithm multiple times according to the input parameters, measures the elapsed time for each execution and writes it in a CSV file.

The **input parameters** of the executable are:
- `INITIAL_VECTOR_SIZE_EXP` and `FINAL_VECTOR_SIZE_EXP`: the first and last size of the vector to loop through, given as the exponents of the desired powers of $2$.
- `INITIAL_THREADS_PER_BLOCK_EXP` and `FINAL_THREADS_PER_BLOCK_EXP`: the first and last number of threads per block to loop through, given as the exponents of the desired powers of $2$.
- `INITIAL_COMPARATORS_PER_THREAD_EXP` and `FINAL_COMPARATORS_PER_THREAD_EXP`: the first and last number of comparators per thread to loop through, given as the exponents of the desired powers of $2$.
- `N_REPETITIONS`: the number of times to repeat the execution of the algorithm.
- `ALGORITHM`: the algorithm to execute. Valid options are:
  - `A` for `bitonic_sort_global_memory_step_by_step`
  - `B` for `bitonic_sort_global_memory_grouped_steps`
  - `C` for `bitonic_sort_shared_memory_grouped_steps`
  - `D` for `bitonic_sort_shared_memory_grouped_steps_warp`
- `CSV_FILE`: the path to the output CSV file.

The fields stored in the **CSV output file** are:
- `vector_size_exponent`: the size of the vector, given as the exponent of the corresponding power of $2$.
- `vector_size`: the size of the vector.
- `threads_per_block_exponent`: the number of threads per block, given as the exponent of the corresponding power of $2$.
- `threads_per_block`: the number of threads per block.
- `comparators_per_thread_exponent`: the number of comparators per thread, given as the exponent of the corresponding power of $2$.
- `comparators_per_thread`: the number of comparators per thread.
- `values_per_block`: the number of values per block.
- `blocks_per_grid`: the number of blocks per grid.
- `repetition`: the index of the current repetition of the execution of the algorithm.
- `elapsed_time_ms`: the execution time of the algorithm in milliseconds.

In order to ignore the CSV output, pass `/dev/null` as value for the `CSV_FILE` parameter.

Usage **examples** (all algorithms):
- `bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 A csv/global_memory_step_by_step.csv`
- `bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 B csv/global_memory_grouped_steps.csv`
- `bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 C csv/shared_memory_grouped_steps.csv`
- `bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 D csv/shared_memory_grouped_steps_warp.csv`


### GPU libraries implementations

The executables `thrust_sort`, `cub_merge_sort` and `cub_radix_sort` run the GPU implementation of the sorting algorithm, measure their execution time and then check that the array is sorted correctly.

The **input parameters** of the executable are:
- `VECTOR_SIZE_EXP`: the size of the vector, given as the exponent of the desired power of $2$.

Usage **examples**:
- `bin/thrust_sort 20`
- `bin/cub_merge_sort 25`
- `bin/cub_radix_sort 25`


#### `bitonic_sort_lib_measure.sh` script

The script `script/bitonic_sort_lib_measure.sh` allows to execute the GPU libraries implementations with multiple vector sizes and write the elapsed time for each execution in a CSV file.

The **input parameters** of the script are:
- `INITIAL_VECTOR_SIZE_EXP` and `FINAL_VECTOR_SIZE_EXP`: the first and last size of the vector to loop through, given as the exponents of the desired powers of $2$.
- `N_REPETITIONS`: the number of times to repeat the execution of the algorithm.
- `EXEC_PATH`: the path to the executable that needs to be run.

The script prints the elapsed time for each execution in CSV format to the standard output, so it can be redirected to a `.csv` file.

The fields printed in **CSV output format** are:
- `vector_size_exponent`: the size of the vector, given as the exponent of the corresponding power of $2$.
- `repetition`: the index of the current repetition of the execution of the algorithm.
- `elapsed_time_ms`: the execution time of the algorithm in milliseconds.

Usage **examples**:
- `script/bitonic_sort_lib_measure.sh 11 23 5 bin/thrust_sort > csv/thrust_sort.csv`
- `script/bitonic_sort_lib_measure.sh 11 30 5 bin/cub_merge_sort > csv/cub_merge_sort.csv`
- `script/bitonic_sort_lib_measure.sh 11 30 5 bin/cub_radix_sort > csv/cub_radix_sort.csv`

*Note*: if used with the `script/sbatch_run_command.sh` script, remember to put the previous commands in quotes, otherwise the CSV file will contain the output from `sbatch`, not from the script. Example: `sbatch script/sbatch_run_command.sh 'script/bitonic_sort_lib_measure.sh 11 23 5 bin/thrust_sort > csv/thrust_sort.csv'`



