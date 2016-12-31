/* Author: Oleksandr Borysov
*  Part 2
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SRAND 32425
#define TYPE double
#define MPI_DATA_TYPE MPI_DOUBLE

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


int main(int argc, char* argv[]) {
   srand(SRAND);
   char *p1 = "base_i_vector_n.dat";
   char *p2 = "random_vector_n.dat";
   char *p3 = "id_matrix_m_n.dat";
   char *p4 = "random_matrix_m_n.dat";
   int i, m, n;
   printf("Type n:\n");
   fflush(stdout);
   scanf("%d",&n);
   printf("Type m:\n");
   fflush(stdout);
   scanf("%d",&m);
   printf("Type i:\n");
   fflush(stdout);
   scanf("%d",&i);
   TYPE* v1 = (TYPE*) base_i_vector_n(i, n);
   TYPE* v2 = (TYPE*) random_vector_n(n);
   TYPE** matr1 = (TYPE**) id_matrix_m_n(m, n);
   TYPE** matr2 = (TYPE**) random_matrix_m_n(m, n);
   store_vector(p1, n, sizeof(TYPE), v1);
   store_vector(p2, n, sizeof(TYPE), v2);
   store_matrix(p3, m, n, sizeof(TYPE), matr1);
   store_matrix(p4, m, n, sizeof(TYPE), matr2);
   vector_free(v1);
   vector_free(v2);
   matrix_free(matr1, m);
   matrix_free(matr2, m);


   TYPE *v3 = (TYPE*) vector_alloc(m, sizeof(TYPE));
   printf("Read file base_i_vector_n.dat\n");
   fflush(stdout);
   read_vector(p1, sizeof(TYPE), &n, &v1);
   print_vector_lf(v1, MPI_DATA_TYPE, n); 
   printf("\nRead file random_vector_n.dat\n");
   fflush(stdout);
   read_vector(p2, sizeof(TYPE), &n, &v2);
   print_vector_lf(v2, MPI_DATA_TYPE, n);
   printf("\nRead file id_matrix_m_n.dat\n");
   fflush(stdout);
   read_matrix(p3, sizeof(TYPE), &m, &n, &matr1);
   print_matrix_lf(matr1, MPI_DATA_TYPE, m, n);
   printf("\nRead file random_matrix_m_n.dat\n");
   fflush(stdout);
   read_matrix(p4, sizeof(TYPE), &m, &n, &matr2);
   print_matrix_lf(matr2, MPI_DATA_TYPE, m, n);
   printf("Seq matr and vector\n");
   fflush(stdout);
   seq_matvec_mult_lf(matr2, v2, m, n, &v3);
   print_vector_lf(v3, MPI_DATA_TYPE, m);
   vector_free(v1);
   vector_free(v2);
   vector_free(v3);
   matrix_free(matr2, m);
   matrix_free(matr1, m);
   MPI_Finalize();
   return 0;
}

void read_row_matrix(char *f, MPI_Datatype dtype, size_t *m, size_t *n, void ***M, MPI_Comm comm) {
    int p, id, size, typeSize;
    MPI_Status st;
    
    MPI_Type_size(dtype, typeSize);
    MPI_Comm_size(comm, &p); 
    MPI_Comm_rank(comm, &id); 

    void ***matr;
    if (id == 0) {
       read_matrix(f, typeSize, m, n, matr);
    }
    MPI_Bcast(m, 1, MPI_INT, 0, comm);
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
	
    size = BLOCK_SIZE(id, p, *m) * *n;
    **M = calloc(size, typeSize);

    if (id == 0) {
	for (int i = 1; i < p; ++i) {
           MPI_Send(*matr[BLOCK_LOW(id, p, *m)], size, typeSize, i, 0, comm);
        }
	memcpy(**M, **matr, size);
	matrix_free(matr, *m);
    } else {
        MPI_Recv(**M, size, typeSize, 0, 0, comm, &st);
    }
}

void read_vector_and_replicate(char *f, MPI_Datatype dtype, size_t *n, void **v, MPI_Comm comm) {
    int p, id, typeSize;
    MPI_Status st;
    
    MPI_Type_size(dtype, typeSize);
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
	matr[i] = vector_alloc(n, size);
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
   for (int i = 0; i < (int) n; ++i) {
      if (type == MPI_INT)
	printf("%d ", ((int *) v)[i]);
      if (type == MPI_DOUBLE)
	printf("%0.2f ", ((double *) v)[i]);
      if (type == MPI_CHAR)
	printf("%c ", ((char *) v)[i]);
   }
}

void read_matrix(char *f, size_t size, size_t *m, size_t *n, void ***M) {	
   FILE *file;
   file = fopen(f, "r");
   fread(m, sizeof(int), 1, file);
   fread(n, sizeof(int), 1, file);
   *M = matrix_alloc(*m, *n, size);
   for (int i = 0; i < (int) *m; ++i) {
       fread((*M)[i], size * *n, 1, file);
   }
}

void store_vector(char *f, size_t n, size_t size, void *v) {
   FILE *file;
   file = fopen(f, "w");
   fwrite(&n, sizeof(int), 1, file);
   fwrite(v, n * size, 1, file);
   fclose(file);
}

void store_matrix(char *f, size_t m, size_t n, size_t size, void **M) {
   FILE *file;
   file = fopen(f, "w");
   fread(&m, sizeof(int), 1, file);
   fread(&n, sizeof(int), 1, file);
   for (int i = 0; i < (int) m; ++i) {
       fwrite(M[i], size * n, 1, file);
   }
   fclose(file);
}

void print_matrix_lf (void **M, MPI_Datatype type, size_t m, size_t n) {
   for (int j = 0; j < (int) m; ++j) {
     for (int i = 0; i < (int) n; ++i) {
       if (type == MPI_INT)
           printf("%d ", ((int**) M)[j][i]);
       if (type == MPI_DOUBLE)
	   printf("%0.2f ", ((double**) M)[j][i]);
       if (type == MPI_CHAR)
	   printf("%c ", ((char**) M)[j][i]);
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

