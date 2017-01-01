/* Author: Oleksandr Borysov
*  Part 2
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SRAND 32425
#define TYPE double
#define MPI_DATA_TYPE MPI_DOUBLE

#define size_t int			// mpicc has problem with mpicc, mpich version 3.2.7

#define BLOCK_LOW(id,p,k) ((id)*(k)/(p))
#define BLOCK_HIGH(id,p,k) (BLOCK_LOW((id)+1,p,k)-1)
#define BLOCK_SIZE(id,p,k) (BLOCK_HIGH(id,p,k)-BLOCK_LOW(id,p,k)+1)
#define BLOCK_OWNER(j,p,k) (((p)*((j)+1)-1)/(k))

void *vector_alloc (size_t n, size_t size );
void vector_free (void *v);
void **matrix_alloc(size_t m, size_t n, size_t size );
void matrix_free (void **M, size_t m);
void store_vector(char *f, size_t n, size_t size, void *v);
void store_matrix(char *f, size_t m, size_t n, size_t size, void **M);
void read_vector(char *f, size_t size, size_t *n, void **v);
void read_matrix(char *f, size_t size, size_t *m, size_t *n, void ***M);

void print_vector_lf (void *v, MPI_Datatype type, size_t n);
void print_matrix_lf (void **M, MPI_Datatype type, size_t m, size_t n);
void seq_matvec_mult_lf(double **M, double *b, size_t m, size_t n, double **c);

TYPE* base_i_vector_n(int i, int n);
TYPE* random_vector_n(int n);
TYPE** id_matrix_m_n(int m, int n);
TYPE** random_matrix_m_n(int m, int n);

void create_mixed_count_disp_arrays(int p, size_t k, int **count, int **disp);
void create_count_disp_arrays(int id,int p,size_t k, int **count, int **disp);


void read_row_matrix(char *f, MPI_Datatype dtype, size_t *m, size_t *n, void ***M, MPI_Comm comm);
void read_vector_and_replicate(char *f, MPI_Datatype dtype, size_t *n, void **v, MPI_Comm comm);
void print_row_vector(void *v, MPI_Datatype type, size_t n, MPI_Comm comm);
void print_row_matrix(void **M, MPI_Datatype type, size_t m, size_t n, MPI_Comm comm);

void copy_matr_part(void **matr, int low, int hight, int n, int typeSize, void **M);

int main(int argc, char* argv[]) {
   srand(SRAND);
   double minTime, maxTime, avrTime, elapsed_time = 0.0;
   MPI_Comm comm = MPI_COMM_WORLD;
   int p, id;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &id); 

   MPI_Barrier(comm);
   elapsed_time = -MPI_Wtime();

   char *p1 = "random_vector_n.dat";
   char *p2 = "random_matrix_m_n.dat";
   char *plotDataFile = "plotData.txt";

   int m, n;
   void* v, *vResult;
   void** matr;
   
   read_row_matrix(p2, MPI_DATA_TYPE, &m, &n, &matr, comm);  
    // allocate matrix
      int rowsNumber = BLOCK_SIZE(id, p, m);
  printf("id=%d rows=%d \n", id, rowsNumber);
  print_matrix_lf(matr, MPI_DATA_TYPE, rowsNumber, n); 
  printf("\n");
  fflush(stdout); 

  /* printf("\nProc id=%d \n", id);
   print_matrix_lf(matr, MPI_DATA_TYPE, m, n);
   fflush(stdout); */
   //read_vector_and_replicate(p1, MPI_DATA_TYPE, &n, &v, comm);
   //print_vector_lf(v, MPI_DATA_TYPE, n);   
   //fflush(stdout);
/*
   vector_free(v);
   vector_free(vResult);
   matrix_free(matr, m);

   elapsed_time += MPI_Wtime();
   MPI_Reduce(&elapsed_time, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
   MPI_Reduce(&elapsed_time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
   MPI_Reduce(&elapsed_time, &avrTime, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

   if (id == 0) {
     avrTime /= p;
     FILE* dataPlotFile;
     dataPlotFile = fopen(plotDataFile, "a");
     fprintf(dataPlotFile, "%d %f %f %f\n", p, maxTime, avrTime, minTime);
     fclose(dataPlotFile);
   } */
   MPI_Finalize();
   return 0;
}

void read_row_matrix(char *f, MPI_Datatype dtype, size_t *m, size_t *n, void ***M, MPI_Comm comm) {
    int p, id, typeSize, size, offset;
    MPI_Status st;
    void **matr;

    MPI_Type_size(dtype, &typeSize);
    MPI_Comm_size(comm, &p); 
    MPI_Comm_rank(comm, &id); 

    if (id == 0) {
       read_matrix(f, typeSize, m, n, &matr);
    }
    
    MPI_Bcast(m, 1, MPI_INT, 0, comm);
    MPI_Bcast(n, 1, MPI_INT, 0, comm);

    void *buff = calloc(*m * *n, typeSize);
    void **array = calloc(*m, sizeof(void*));
    for (int i = 0; i < *m; i++) {
        array[i] = (buff + *n * typeSize * i);
    }
    *M = array;
    
    if (id == 0) {
	for (int i = 1; i < p; ++i) {
           size = BLOCK_SIZE(i, p, *m) * *n; 
           offset = BLOCK_LOW(i, p, *m);
           MPI_Send(&matr[offset][0], size, dtype, i, 0, comm);
        }
        size = (BLOCK_SIZE(id, p, *m) + 1) * *n; 
        offset = BLOCK_LOW(id, p, *m);
        memcpy(&(*M)[0], &matr[offset], size); 
    } else {        
        size = BLOCK_SIZE(id, p, *m) * *n; 
        MPI_Recv(&(*M)[0][0], size, dtype, 0, 0, comm, &st);
    }
  //matrix_free(matr, *m);
}

void copy_matr_part(void **matr, int low, int hight, int n, int typeSize, void **M) {
    for (int i = low, ii = 0; i <= hight; ++i, ++ii) {
       M[ii] = matr[i];
    }
}

void read_vector_and_replicate(char *f, MPI_Datatype dtype, size_t *n, void **v, MPI_Comm comm) {
    int p, id, typeSize;
    MPI_Status st;
    
    MPI_Type_size(dtype, &typeSize);
    MPI_Comm_size(comm, &p); 
    MPI_Comm_rank(comm, &id); 

    if (id == 0) {
       read_vector(f, typeSize, n, v);
    } else {	
       *v = calloc(*n, typeSize);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
    MPI_Bcast(*v, *n, dtype, 0, comm);
}

void print_row_vector(void *v, MPI_Datatype type, size_t n, MPI_Comm comm) {
    
}

void print_row_matrix(void **M, MPI_Datatype type, size_t m, size_t n, MPI_Comm comm) {
    
}

void create_count_disp_arrays(int id, int p, size_t k, int **count, int **disp) {
    int size = BLOCK_SIZE(id, p, k);
    (*disp)[0] = 0;
    (*count)[0] = size;
    for (int i = 1; i < p; ++i) {
	(*disp)[i] = (*disp)[i-1] + (*count)[i-1];
	(*count)[i] = size;
    }
}

void create_mixed_count_disp_arrays(int p, size_t k, int **count, int **disp) {
    (*count)[0] = BLOCK_SIZE(0, p, k);
    (*disp)[0] = 0;

    for (int i = 1; i < p; ++i) {
	(*disp)[i] = (*disp)[i-1] + (*count)[i-1];
	(*count)[i] = BLOCK_SIZE(i, p, k);
    }
}

TYPE* base_i_vector_n(int i, int n) {
    TYPE *v = (TYPE*) vector_alloc(n, sizeof(TYPE));
    v[i] = 1;
    return v;
}

TYPE* random_vector_n(int n) {
    TYPE *v = (TYPE*) vector_alloc(n, sizeof(TYPE));
    for (int i = 0; i < n; i++)
       v[i] = rand() % 255;
    return v;
}

TYPE** id_matrix_m_n(int m, int n) {
    TYPE **matr = (TYPE**) matrix_alloc(m, n, sizeof(TYPE));
    if (m < n) 
      for (int i = 0; i < m; ++i)
         matr[i][i] = 1;
    else for (int i = 0; i < n; ++i)
            matr[i][i] = 1;
    return matr;
}

TYPE** random_matrix_m_n(int m, int n) {
    TYPE **matr = (TYPE**) matrix_alloc(m, n, sizeof(TYPE));
    for (int i = 0; i < m; ++i)
       for (int j = 0; j < n; ++j)
          matr[i][j] = (rand() % 255);
    return matr;
}

void *vector_alloc (size_t n, size_t size) {
    return calloc(n, size);	 
}

void vector_free (void *v) {
   free(v);
}

void **matrix_alloc(size_t m, size_t n, size_t size ) {
    void **matr = calloc(m, sizeof(void*));
    for (int i = 0; i < m; i++) {
	matr[i] = calloc(n, size);
    }
    return matr;	 
}

void matrix_free (void **M, size_t m) {
    for (int i = 0; i < m; i++) {
	free(M[i]);
    }
    free(M);	 
}

void read_vector(char *f, size_t size, size_t *n, void **v)  {
   FILE *file;
   file = fopen(f, "r");
   fread(n, sizeof(int), 1, file);
   *v = vector_alloc(*n, sizeof(TYPE));
   fread(*v, size * *n, 1, file);
   fclose(file);
}

void print_vector_lf (void *v, MPI_Datatype type, size_t n) {
   for (int i = 0; i < n; ++i) {
      if (type == MPI_INT)
	printf("%d ", ((int *) v)[i]);
      if (type == MPI_DOUBLE)
	printf("%0.2f ", ((double *) v)[i]);
      if (type == MPI_CHAR)
	printf("%c ", ((char *) v)[i]);
   }
}

void store_vector(char *f, size_t n, size_t size, void *v) {
   FILE *file;
   file = fopen(f, "w");
   fwrite(&n, sizeof(int), 1, file);
   fwrite(v, n * size, 1, file);
   fclose(file);
}

void read_matrix(char *f, size_t size, size_t *m, size_t *n, void ***M) {	
   FILE *file;
   file = fopen(f, "r");   
   fread(m, sizeof(size_t), 1, file);
   fread(n, sizeof(size_t), 1, file);
   void *buff = calloc(*m * *n, size);
   void **array = calloc(*m, sizeof(void*));
   fread(buff, *m * *n * size, 1, file);
   for (int i = 0; i < *m; i++) {
        array[i] = (buff + *n * size * i);
   }
   *M = array;
   fclose(file);
}

void store_matrix(char *f, size_t m, size_t n, size_t size, void **M) {
   FILE *file;
   file = fopen(f, "w");
   fwrite(&m, sizeof(size_t), 1, file);
   fwrite(&n, sizeof(size_t), 1, file);
   for (int i = 0; i < m; ++i) {
       fwrite(M[i], size * n, 1, file);
   }
   fclose(file);
}
 
void print_matrix_lf (void **M, MPI_Datatype type, size_t m, size_t n) {
   for (int i = 0; i < m; ++i) {
     for (int j = 0; j < n; ++j) {
       if (type == MPI_INT)
           printf("%d ", ((int**) M)[i][j]);
       if (type == MPI_DOUBLE)
	   printf("%0.2f ", ((double**) M)[i][j]);
       if (type == MPI_CHAR)
	   printf("%c ", ((char**) M)[i][j]);
     }
     printf("\n");
   }
}

void seq_matvec_mult_lf(double **M, double *b, size_t m, size_t n, double **c) {
    for (int i = 0; i < m; ++i) {
       (*c)[i] = 0;
       for (int j = 0; j < n; ++j) {
          (*c)[i] += M[i][j] * b[j];
       }
    }
}

