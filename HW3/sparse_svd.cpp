
#include <stdio.h>
#include <math.h>
#include <algorithm>

extern "C"
{
    void dgesvd_(char *jobu, char *jobvt, int *m, int *n,
                 double *A, int *lda, double *S, double *U,
                 int *ldu, double *VT, int *ldvt, double* work,
                 int *lwork, int *info);

    void dgemm_(char *transa, char *transb, int *m, int * n,
                int *k, double *alpha, double *a, int *lda, 
                double *b, int *ldb, double *beta, double *c, int *ldc);
}
void print_matrix(char const* label, double *A, int m, int n) {
    int i, j;
#define A(i,j) (*((double *)A + i + j * m))
    printf("%s\n",label);
    for (i=0; i<m; i++){
        for (j=0; j<n; j++) 
            printf("%20.10lf ",A(i,j));
        printf("\n");
    }
    printf("\n");
}

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

void transpose(double* m, size_t h, size_t w)
{
    for (unsigned start = 0; start <= w * h - 1; ++start)
    {
        unsigned next = start;
        unsigned i = 0;
        do
        {
            ++i;
            next = (next % w) * h + next / w;
        } while (next > start);

        if (next >= start && i != 1)
        {
            const double tmp = m[start];
            next = start;
            do
            {
                i = (next % w) * h + next / w;
                m[next] = (i == start) ? tmp : m[i];
                next = i;
            } while (next > start);
        }
    }
}

void smat_mult(double* A,
               double* U, double* D, double* V,
               size_t m, size_t k, size_t n)
{
    for(size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < k; ++j) {
            U[i  + j * m] = U[i  + j * m] * D[j];
        }
    }

    for(size_t i = 0; i < k; ++i) {
        for(size_t j = 0; j < n; ++j) {
            A[i + j * m] = 0;
            for(size_t k = 0; k < n; ++k) {
                A[i + j * m] += U[i + k * m] * V[k + j * m];
            }
        }
    }
}

#define M 4
#define K 4
#define N 4
#define LWORK max(3 * min(M,N) + max(M,N), 5 * min(M,N))

int main() {
    int m = M, n = N, lwork = LWORK, info;
    int k = K;
    double A[M * N] = {1.0,2.0,3.0,4.0,5.0,6.0,
                       7.0,8.0,9.0,10.0,11.0,12.0,
                       13.0,14.0,15.0,16.0,/*17.0,18.0,
                       19.0,20.0, 21.0,22.0,23.0,24.0*/};
    
    print_matrix("A =",A,N,M);
    transpose(A,N,M);
    print_matrix("A =",A,M,N);

    double S[N];
    double U[M * M];
    double VT[N * N];
    double work[LWORK];
    char job = 'a';

    dgesvd_(&job, &job, &m, &n, A, &m, S, U, &m, VT, &n, work, &lwork, &info);

    if (info != 0)
        fprintf(stderr,"DGESVD returned info = %d\n",info);

    print_matrix("U =",U,M,K);
    print_matrix("Sigma =",S,K,1);
    print_matrix("VT =",VT,K,N);

    smat_mult(A, U, S, VT, m, k, n);
    print_matrix("A =", A, M, N);
    return(0);
}
