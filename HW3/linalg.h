
#ifndef _HW3_LINALG_H_
#define _HW3_LINALG_H_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <tuple>

extern "C"
{
    void dgesvd_(char *jobu,
                 char *jobvt,
                 int *m, int *n,
                 double *A, int *lda,
                 double *S,
                 double *U, int *ldu,
                 double *VT, int *ldvt,
                 double* work, int *lwork, int *info);

    void dgemm_(char* TRANSA, char* TRANSB,
                int* M, int* N, int* K,
                double* ALPHA, double* A, int* 	LDA,
                double* B, int* LDB, double* BETA,
                double* C, int* LDC);	

    void dgemv_(char* TRANSA,
                int* M, int* N, 
                double* ALPHA, double const* A, int* LDA,
                double const* X, int* INCX, double* BETA,
                double* Y, int* INCY);	

    double dnrm2_(int* n, double const* x, int* incx);
}

namespace linalg
{
    class dense_matrix
    {
        size_t _m;
        size_t _n;
        double* _data;

    public:
        inline
        dense_matrix()
            :_m(0), _n(0), _data(nullptr)
        {}

        inline
        dense_matrix(size_t m, size_t n)
            : _m(m),
              _n(n),
              _data(static_cast<double*>(
                        malloc(m * n * sizeof(double))))
        {
            memset(_data, 0.0f, sizeof(double) * m * n);
        }

        inline
        dense_matrix(dense_matrix const& other)
            : _m(other._m),
              _n(other._n),
              _data(static_cast<double*>(
                        malloc(_m * _n * sizeof(double))))
        {
            memcpy(_data, other._data, sizeof(double) * _m * _n);
        }

        inline
        dense_matrix(dense_matrix&& other)
            : _m(other._m),
              _n(other._n),
              _data(other._data)
        {
            other._data = nullptr;
            other._m = 0;
            other._n = 0;
        }

        inline dense_matrix&
        operator=(dense_matrix const& other) noexcept
        {
            if(_data != nullptr)
                free(_data);

            _m = other._m;
            _n = other._n;
            _data = static_cast<double*>(
                malloc(_m * _n * sizeof(double)));
            memcpy(_data, other._data, _m * _n * sizeof(double));
            return *this;
        }

        inline dense_matrix&
        operator=(dense_matrix&& other) noexcept
        {
            if(_data != nullptr)
                free(_data);

            _data = other._data;
            _m = other._m;
            _n = other._n;
            other._data = nullptr;
            other._m = 0;
            other._n = 0;
            return *this;
        }

        inline void
        normalize_cols()
        {
            for(size_t j = 0; j < _n; ++j)
            {
                auto begin = &_data[j * _m];
                auto end = &_data[(j + 1) * _m];
                double eps = std::numeric_limits<double>::epsilon();
                int total = std::accumulate(begin, end, 0);
                std::transform(begin, end, begin,
                               [total, eps](double elem)
                               {return elem / (total + eps); } );
            }
        }

        inline double
        row_norm(size_t i)
        {
            int n = _n;
            int offset = _m;
            return dnrm2_(&n, &_data[i], &offset);
        }

        inline double 
        operator()(size_t i, size_t j) const
        {
            return _data[i + j * _m];
        }

        inline double&
        operator()(size_t i, size_t j)
        {
            return _data[i + j * _m];
        }
        
        inline double*
        data() const noexcept
        {
            return _data;
        }

        inline std::pair<size_t, size_t>
        shape() const noexcept
        { return {_m, _n}; }

        inline ~dense_matrix() noexcept
        {
            if(_data != nullptr)
                free(_data);
        }

    };

    inline std::tuple<dense_matrix, std::vector<double>, dense_matrix>
    svd(dense_matrix const& mat)
    {
        int m = mat.shape().first;
        int n = mat.shape().second;
        auto U = dense_matrix(m, m);
        auto Vt = dense_matrix(n, n);
        auto D = std::vector<double>(n);
        int lwork = std::max(3 * std::min(m,n) + std::max(m,n),
                             5 * std::min(m,n));
        auto work = std::vector<double>(lwork);
        int info = 0;

        char job = 'a';
        dgesvd_(&job, &job, &m, &n, mat.data(), &m, D.data(), U.data(),
                &m, Vt.data(), &n, work.data(), &lwork, &info);
        return {U, D, Vt};
    }

    inline std::vector<double>
    operator*(double a, std::vector<double> const& v)
    {
        std::vector<double> result(v);
        std::transform(v.begin(), v.end(), result.begin(),
                       [a](double elem){ return a * elem; });
        return result;
    }

    inline dense_matrix
    operator*(dense_matrix const& A, dense_matrix const& B)
    {
        assert(A.shape().second == B.shape().first);
        char t = 'n';
        double alpha = 1;
        double beta = 0;
        int m = A.shape().first;
        int k = A.shape().second;
        int n = B.shape().first;
        auto C = dense_matrix(m, n);

        dgemm_(&t, &t,
               &m, &n, &k,
               &alpha, A.data(), &m,
               B.data(), &k, &beta,
               C.data(), &m);
        return C;
    }

    inline void
    print_matrix(char const* label, dense_matrix const& mat) {
        int m = mat.shape().first;
        int n = mat.shape().second;

        int i, j;
        printf("%s\n",label);
        for (i=0; i<m; i++){
            for (j=0; j<n; j++) 
                printf("%20.10lf ", mat(i,j));
            printf("\n");
        }
        printf("\n");
    }

    inline dense_matrix
    approx_matscaling(dense_matrix const& U,
                      std::vector<double> const& D,
                      size_t approx_rank_k)
    {
        size_t m = U.shape().first;
        auto C = dense_matrix(m, approx_rank_k);

        for(size_t j = 0; j < approx_rank_k; ++j) {
            for(size_t i = 0; i < m; ++i)
                C(i, j) = U(i, j) * D[j];
        }
        return C;
    }

    inline double
    norm(std::vector<double> const& vec)
    {
        int n = vec.size();
        int offset = 1;
        return dnrm2_(&n, vec.data(), &offset);
    }

    inline std::vector<double>
    gemv(double alpha,
         dense_matrix const& A,
         std::vector<double> const& v)
    {
        std::vector<double> result(v.size());
        char t = 'N';
        int m = A.shape().first;
        int n = A.shape().second;
        double beta = 0;
        int incx = 1;
        int incy = 1;
        dgemv_(&t, &m, &n, 
               &alpha, A.data(), &m,
               v.data(), &incx, &beta,
               result.data(), &incy);	
        return result;
    }
}

#endif
