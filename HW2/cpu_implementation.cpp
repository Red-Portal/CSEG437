
#include "cpu_implementation.h"
#include "my_OpenCL_util.h"
#include "parameters.h"
#include "thread_pool.h"
#include "ode_solver.h"

#include <cstdlib>
#include <cmath>
#include <iostream>

struct point
{
    float x;
    float y;
    float z;
    float t;
};

point pos_in[NUM_PARTICLES_X * NUM_PARTICLES_Y];
point vel_in[NUM_PARTICLES_X * NUM_PARTICLES_Y];

point pos_out[NUM_PARTICLES_X * NUM_PARTICLES_Y];
point vel_out[NUM_PARTICLES_X * NUM_PARTICLES_Y];

void copy_to_host(cl_command_queue queue,
                  cl_mem* buf_pos,
                  size_t pos_n,
                  cl_mem* buf_vel,
                  size_t vel_n)
{
    CHECK_ERROR_CODE(
        clEnqueueReadBuffer(queue, *buf_pos, CL_FALSE, 0,
                            sizeof(float)*pos_n, (void*)&pos_in, 0, NULL, NULL));
    CHECK_ERROR_CODE(
        clEnqueueReadBuffer(queue, *buf_vel, CL_FALSE, 0,
                            sizeof(float)*vel_n, (void*)&vel_in, 0, NULL, NULL));
}

void copy_to_device(cl_command_queue queue,
                    cl_mem* buf_pos,
                    size_t pos_n,
                    cl_mem* buf_vel,
                    size_t vel_n)
{
    CHECK_ERROR_CODE(
        clEnqueueWriteBuffer(queue, *buf_pos, CL_FALSE, 0,
                             sizeof(float)*pos_n, (void const*)&pos_out[0], 0, NULL, NULL));
    CHECK_ERROR_CODE(
        clEnqueueWriteBuffer(queue, *buf_vel, CL_FALSE, 0,
                             sizeof(float)*vel_n, (void const*)&vel_out[0], 0, NULL, NULL));
}

inline size_t index(size_t x, size_t y)
{ return x * NUM_PARTICLES_Y; }

inline float l2norm(point const* arr)
{
    float sum = 0;
    sum += arr->x * arr->x;
    sum += arr->y * arr->y;
    sum += arr->z * arr->z;
    return sqrt(sum);
}

inline void vecadd(point const* a, point const* b, point* c)
{
    c->x = a->x + b->x;
    c->y = a->y + b->y;
    c->z = a->z + b->z;
}

inline void vecsub(point const* a, point const* b, point* c)
{
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

inline void scalarmult(float a, point const* v, point* c)
{
    c->x = a * v->x;
    c->y = a * v->y;
    c->z = a * v->z;
}

inline void force_single_direction(float constant,
                                   float rest_length,
                                   point const& current,
                                   point const& other,
                                   point* result)
{
    point dist;
    vecsub(&current, &other, &dist);
    float norm = l2norm(&other);
    float coef = constant * (norm - rest_length) / norm;
    scalarmult(coef, &dist, result);
}

inline void spring_force(size_t i, size_t j, 
                         point* p,
                         float spring_const,
                         float RestLengthHoriz,
                         float RestLengthVert,
                         float RestLengthDiag,
                         point* force)
{
    point result= {0,0,0,0};
    point temp;
    point& current = p[index(i, j)];

    if(j != 0)
    {
        force_single_direction(spring_const,
                               RestLengthVert,
                               current,
                               p[index(i, j-1)],
                               &temp);
        vecadd(&result, &temp, &result);
    }

    if(j != (NUM_PARTICLES_Y - 1))
    {
        force_single_direction(spring_const,
                               RestLengthVert,
                               current,
                               p[index(i-1, j+1)],
                               &temp);
        vecadd(&result, &temp, &result);
    }

    if(i != 0)
    {
        force_single_direction(spring_const,
                               RestLengthHoriz,
                               current,
                               p[index(i-1, j)],
                               &temp);
        vecadd(&result, &temp, &result);

        if(j != 0)
        {
            force_single_direction(spring_const,
                                   RestLengthDiag,
                                   current,
                                   p[index(i-1, j-1)],
                                   &temp);
            vecadd(&result, &temp, &result);
        }
        if(j != (NUM_PARTICLES_Y - 1))
        {
            force_single_direction(spring_const,
                                   RestLengthDiag,
                                   current,
                                   p[index(i-1, j+1)],
                                   &temp);
            vecadd(&result, &temp, &result);
        }
    }
    if(i != (NUM_PARTICLES_X - 1))
    {
        force_single_direction(spring_const,
                               RestLengthHoriz,
                               current,
                               p[index(i+1, j)],
                               &temp);

        vecadd(&result, &temp, &result);
        if(j != 0)
        {
            force_single_direction(spring_const,
                                   RestLengthDiag,
                                   current,
                                   p[index(i+1, j+1)],
                                   &temp);
            vecadd(&result, &temp, &result);
        }
        if(j != (NUM_PARTICLES_Y - 1))
        {
            force_single_direction(spring_const,
                                   RestLengthDiag,
                                   current,
                                   p[index(i+1, j-1)],
                                   &temp);
            vecadd(&result, &temp, &result);
        }
    }

    *force = result;
}

void cloth_position_host(float const* Gravity, // float3,
                         float ParticleMass,
                         float ParticleInvMass,
                         float SpringK,
                         float RestLengthHoriz,
                         float RestLengthVert,
                         float RestLengthDiag,
                         float DeltaT,
                         float DampingConst)
{
    parallel_for_loop(
        0, NUM_PARTICLES_X,
        [=](size_t i)
        {
            size_t start_idx = NUM_PARTICLES_Y * i;

            for(size_t j = 0; j < NUM_PARTICLES_Y; ++j)
            {
                size_t idx = start_idx + j;
                point static_offset = {0.05, 0.05, 0.05, 0};
                vecsub(&pos_in[idx], &static_offset, &pos_out[idx]);
            }
        });

    // parallel_for_loop(
    //     0, NUM_PARTICLES_X,
    //     [=](size_t i)
    //     {
    //         size_t start_idx = NUM_PARTICLES_Y * i;

    //         for(size_t j = 0; j < NUM_PARTICLES_Y; ++j)
    //         {
    //             size_t idx = start_idx + j;

    //             point spring_f;
    //             spring_force(i, j, 
    //                          &pos_in[idx],
    //                          SpringK,
    //                          RestLengthHoriz,
    //                          RestLengthVert,
    //                          RestLengthDiag,
    //                          &spring_f);

    //             auto update = [=](float const* y,
    //                               float t,
    //                               float* dy){
                    
                    
                    
    //             };

    //             pos_out[idx] = pos_in[idx];
    //             solver_euler(std::move(update),
    //                          reinterpret_cast<float*>(&pos_out),
    //                          0.0f,
    //                          DeltaT,
    //                          iter);
    //         }
    //     });
}
