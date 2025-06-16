#!/bin/bash

if [[ $# -ne 4 ]]; then
	echo "ERROR: Illegal number of parameters"
	echo "USAGE: $0 INITIAL_VECTOR_SIZE_EXP FINAL_VECTOR_SIZE_EXP N_REPETITIONS EXEC_PATH"
	exit 1
fi

INITIAL_VECTOR_SIZE_EXP=$1
FINAL_VECTOR_SIZE_EXP=$2
N_REPETITIONS=$3
EXEC_PATH="$4"

echo "vector_size_exponent,repetition,elapsed_time_ms"

for vector_size_exponent in $(seq $INITIAL_VECTOR_SIZE_EXP $FINAL_VECTOR_SIZE_EXP); do
	for repetition in $(seq 1 $N_REPETITIONS); do
		ELAPSED_TIME_MILLIS=$($EXEC_PATH $vector_size_exponent | grep Elapsed | cut -d ' ' -f 3)
		echo $vector_size_exponent,$repetition,$ELAPSED_TIME_MILLIS
	done
done

