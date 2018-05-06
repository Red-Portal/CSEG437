
#include "cpu_implementation.h"
#include "my_OpenCL_util.h"
#include "parameters.h"

#include <cstdlib>

float pos_in[NUM_PARTICLES_X * NUM_PARTICLES_Y * 4];
float vel_in[NUM_PARTICLES_X * NUM_PARTICLES_Y * 4];

float pos_out[NUM_PARTICLES_X * NUM_PARTICLES_Y * 4];
float vel_out[NUM_PARTICLES_X * NUM_PARTICLES_Y * 4];

void copy_to_host(cl_command_queue queue,
                  cl_mem* buf_pos,
                  size_t pos_n,
                  cl_mem* buf_vel,
                  size_t vel_n)
{
    CHECK_ERROR_CODE(
        clEnqueueReadBuffer(queue, *buf_pos, CL_FALSE, 0,
                            sizeof(float)*pos_n, (void*)&pos_in[0], 0, NULL, NULL));
    CHECK_ERROR_CODE(
        clEnqueueReadBuffer(queue, *buf_vel, CL_FALSE, 0,
                            sizeof(float)*vel_n, (void*)&vel_in[0], 0, NULL, NULL));
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

void cloth_position_host(float* Gravity, // float3,
                         float ParticleMass,
                         float ParticleInvMass,
                         float SpringK,
                         float RestLengthHoriz,
                         float RestLengthVert,
                         float RestLengthDiag,
                         float DeltaT,
                         float DampingConst)
{
    
}
