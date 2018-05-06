#ifndef _CPU_IMPLEMENTATION_H_
#define _CPU_IMPLEMENTATION_H_

#include <cstdlib>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

void copy_to_host(cl_command_queue queue,
                  cl_mem* buf_pos,
                  size_t pos_n,
                  cl_mem* buf_vel,
                  size_t vel_n);

void copy_to_device(cl_command_queue queue,
                    cl_mem* buf_pos,
                    size_t pos_n,
                    cl_mem* buf_vel,
                    size_t vel_n);

void cloth_position_host(float* Gravity, // float3,
                         float ParticleMass,
                         float ParticleInvMass,
                         float SpringK,
                         float RestLengthHoriz,
                         float RestLengthVert,
                         float RestLengthDiag,
                         float DeltaT,
                         float DampingConst);

#endif
