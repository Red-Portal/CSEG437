
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
    void dgesvd_(char *jobu, char *jobvt, int *m, int *n,
                 double *A, int *lda, double *S, double *U,
                 int *ldu, double *VT, int *ldvt, double* work,
                 int *lwork, int *info);
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
    svd(dense_matrix const& mat, size_t order_k)
    {
        size_t m = mat.shape().first;
        size_t n = mat.shape().second;
        assert(order_k <= mat.shape().second);
        auto U = dense_matrix(m, m);
        auto Vt = dense_matrix(order_k, order_k);
        auto D = std::vector<double>(order_k);

        char job = 'a';
        dgesvd_(&job, &job, &m, &n, mat.data(), int *lda, double *S, double *U,
                int *ldu, double *VT, int *ldvt, double* work,
                int *lwork, int *info);
    }
}

#endif
