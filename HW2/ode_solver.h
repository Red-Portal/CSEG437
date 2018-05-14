
#ifndef _ODE_SOLVER_H_
#define _ODE_SOLVER_H_

#include <cstdlib>

template<typename F, typename Type, size_t N>
inline void solver_euler(F&& f,
                         float* y,
                         Type start_time,
                         Type dt,
                         size_t iter)
{
    float dy[N];
    Type t = start_time;
    for(size_t i = 0; i < iter; ++i)
    {
        f(y, t, dy);

        for(size_t j = 0; j < N; ++j)
            y[j] += dy[j] * dt;

        t += dt;
    }
}

template<typename F, typename Type, size_t N>
inline void solver_rk2(F&& f,
                       float* y,
                       Type start_time,
                       Type dt,
                       size_t iter)
{
    float dy[N];
    float predictor[N];
    Type t = start_time;
    for(size_t i = 0; i < iter; ++i)
    {
        f(y, t, dy);

        for(size_t j = 0; j < N; ++j)
            predictor.value[j] = y[j] + dy[j] * dt;
        f(predictor, t + dt, predictor);

        for(size_t j = 0; j < N; ++j)
            y[j] += (dt / 2) * ((predictor[j] * dt) + (dy[j] * dt));

        t += dt;
    }
}


#endif
