
#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

/******************************************************************************************************/
// For cloth simulation
#define PRIM_RESTART 0xffffff

#define GL_INTEROP


const bool use_cpu = true;

const int MAX_THREADS = 4;
const int WORKGROUP_SIZE_X = 16;
const int WORKGROUP_SIZE_Y = 8;

const int NUM_PARTICLES_X = 64;
const int NUM_PARTICLES_Y = 64;

const float CLOTH_SIZE_X = 4.0f;
const float CLOTH_SIZE_Y = 3.0f;

const float PARTICLE_MASS = 0.015;
const float PARTICLE_INV_MASS = 1.0 / PARTICLE_MASS;

const size_t INTERVAL = 100; 
const float SPRING_K = 500.0;
const float GRAVITY[4] = { 0, -9.80665 , 0 };
const float DAMPING_CONST = 0.01;

// Try to use different NUM_ITERs for each tested numerical method.

/******************************************************************************************************/

#endif
