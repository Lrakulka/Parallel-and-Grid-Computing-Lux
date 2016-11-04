/**
* Author Oleksandr Borysov
* Lab Assignment MPI
* 11/1/2016
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <mpi.h>

/* Return 1 if ’i’th bit of ’n’ is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)
#define EXTRACT_BIT_CHAR(n, i) ((n & (1 << i)) ? '1': '0')

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

char *int_to_binary(int x) {
    unsigned short bits = sizeof(int) * 8; // bits in int
    char *b = (char*)calloc(bits + 1, sizeof(char));
    for (unsigned short i = 0; i < bits; i++) {
       b[i] = EXTRACT_BIT_CHAR(x, i);
    }
    b[bits] = '\0';
    return b;
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
    char *result;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    for (unsigned register int input = id; input < INPUT_POSSIBILITIES; input += p) {
       if (check_circuit(input)) {
         result = int_to_binary(input);
         xprintf("Found solution '%s'\n", result);
         free(result);
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    xprintf("done! \n");
    MPI_Finalize();
    return 0;
}


