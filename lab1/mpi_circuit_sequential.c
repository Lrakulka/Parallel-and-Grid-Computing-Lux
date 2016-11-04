/**
* Author Oleksandr Borysov
* Lab Assignment MPI
* 11/1/2016
*/

#include "circuit.c"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

const unsigned int INPUT_POSSIBILITIES = 65536;  // 2^16
unsigned int id = 0; // id of process

/**
 * Redefinition of the printf to include the buffer flushing
 */
void xprintf(char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[Node %i] ", id);
    vprintf(format, args);
    fflush(stdout);
}

char *int_to_binary(int x) {
    unsigned short bits = sizeof(int) * 8; // bits in int
    char *b = (char*)calloc(bits, sizeof(char));
    for (unsigned short i = 0; i < bits; i++) {
       b[i] = EXTRACT_BIT(x, i) ? '1' : '0';
    }

    return b;
}

int main(int argc, char *argv[]) {
    unsigned int p; // MPI specific: number of processors
    unsigned int solution = 0; // Count solutions
    unsigned int countSolutions = 0;
    char *result;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    for (unsigned register int input = id; input < INPUT_POSSIBILITIES; input += p) {
       if (check_circuit(input)) {
         result = int_to_binary(input);
         xprintf("Found solution '%s'\n", result);
         free(result);
	 solution++;
       }
    }
    MPI_Reduce(&solution, &countSolutions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!id) {
       xprintf("done! Found %d solutions\n", countSolutions);
    } else {
       xprintf("done! \n");
    }
    MPI_Finalize();
    return 0;
}


