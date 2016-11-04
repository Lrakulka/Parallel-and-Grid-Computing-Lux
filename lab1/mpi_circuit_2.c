/**
* Author Oleksandr Borysov
* Lab Assignment MPI
* 11/1/2016
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>


/* Return 1 if ’i’th bit of ’n’ is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

const unsigned int INPUT_POSSIBILITIES = 65536;  // 2^16
unsigned int id = 0; // id of process

int check_circuit(int input) {
    int v[16], i;
    for (i = 0; i < 16; i++) v[i] = EXTRACT_BIT(input,i);
    return ((v[0] || v[1])    && (!v[1] || !v[3]) && (v[2] || v[3])  &&
            (!v[3] || !v[4])  && (v[4] || !v[5])  && (v[5] || !v[6]) &&
            (v[5] || v[6])    && (v[6] || !v[15]) && (v[7] || !v[8]) &&
            (!v[7] || !v[13]) && (v[8] || v[9])   && (v[8] || !v[9]) &&
            (!v[9] || !v[10]) && (v[9] || v[11])  && (v[10] || v[11])&&
            (v[12] || v[13])  && (v[13] || !v[14])&& (v[14] || v[15]));
}

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

int main(int argc, char *argv[]) {
    unsigned int p; // MPI specific: number of processors
    unsigned int countSolutions = 0;
    unsigned int solutions = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    for (unsigned register int input = id; input < INPUT_POSSIBILITIES; input += p) {
       if (check_circuit(input)) {
         solutions++;
       }
    }

    MPI_Reduce(&solutions, &countSolutions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!id) {
       xprintf("done! Found %d solutions\n", countSolutions);
    } else {
       xprintf("done! \n");
    }
    MPI_Finalize();
    return 0;
}
