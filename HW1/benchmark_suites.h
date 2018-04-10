#ifndef _BENCHMARK_SUITES_H_
#define _BENCHMARK_SUITES_H_

#include <stdlib.h>
#include <stdio.h>
#include "personal_utils.h"
#include "opencl_wrapper.h"

inline bool check_result(float gpu, float cpu)
{
    float eps = 0.001;
    bool is_correct = abs(gpu - cpu) < eps;
    if(!is_correct)
    {
        printf("-- result not matching!!\n");
        printf("-- cpu result: %f , gpu result: %f\n", cpu, gpu);
    }
    return is_correct;
}

struct global_1d
{
    size_t _problem_size;
    size_t _group_dim;
    size_t _group_num;
    float* _input;
    float* _partial;
    float _gpu;
    float _cpu;

    opencl_kernel* _kernel;
    cl_context _context;
    cl_command_queue _cmd_queue;

    opencl_buffer<float> _partial_buffer;
    opencl_buffer<float> _input_buffer;

    inline global_1d(cl_context context,
                     cl_command_queue cmd_queue,
                     opencl_kernel* kernel)
        :_problem_size(0),
         _group_dim(0),
         _group_num(0),
         _input(NULL),
         _partial(NULL),
         _gpu(0),
         _cpu(0),
         _kernel(kernel),
         _context(context),
         _cmd_queue(cmd_queue),
         _partial_buffer(),
         _input_buffer() {} 

    inline void init(size_t* problem_size)
    {
        _problem_size = *problem_size;
        _input = (float*)malloc(sizeof(float) * _problem_size);
        fill_random(_input, _input + _problem_size, -1.0f, 1.0f);

        _input_buffer.init(_context, mem_flag::read, _problem_size);
        _partial_buffer = opencl_buffer<float>();

    }

    inline void work_dimensions(size_t* group_dim)
    {
        _group_dim = group_dim[0];
        _group_num = (size_t)ceil((float)_problem_size / _group_dim);

        _partial = (float*)malloc(sizeof(float) * _group_num);
        _partial_buffer.init(_context, mem_flag::read_write, _group_num);
    }

    inline void prepare()
    {
        _gpu = 0;
        fill(_partial, _partial + _group_num, 0.0f);

        _input_buffer.enqueue_write(_cmd_queue, _input);
        _partial_buffer.enqueue_write(_cmd_queue, _partial);
        clFinish(_cmd_queue);
    }

    inline void run_gpu()
    {
        _kernel->enqueue_args(&_input_buffer, &_partial_buffer);
        _kernel->enqueue_run(_cmd_queue, 1, &_problem_size, &_group_dim, NULL);

        _partial_buffer.enqueue_read(_cmd_queue, _partial);
        clFinish(_cmd_queue);  

        _gpu = kahan_reduce(_partial, _partial + _group_num, 0.0f);
    }

    inline void run_cpu()
    {
        _cpu = kahan_reduce(_input, _input + _problem_size, 0.0);
    }

    inline bool check_result()
    {
        return ::check_result(_gpu, _cpu);
    }

    inline void teardown()
    {
        return;
    }

    inline ~global_1d()
    {
        free(_partial);
        free(_input);
    }
};

struct local_1d
{
    size_t _problem_size;
    size_t _group_dim;
    size_t _group_num;
    float* _input;
    float* _partial;
    float _gpu;
    float _cpu;

    opencl_kernel* _kernel;
    cl_context _context;
    cl_command_queue _cmd_queue;

    opencl_buffer<float> _partial_buffer;
    opencl_buffer<float> _input_buffer;

    inline local_1d(cl_context context,
                    cl_command_queue cmd_queue,
                     opencl_kernel* kernel)
        :_problem_size(0),
         _group_dim(0),
         _group_num(0),
         _input(NULL),
         _partial(NULL),
         _gpu(0),
         _cpu(0),
         _kernel(kernel),
         _context(context),
         _cmd_queue(cmd_queue),
         _partial_buffer(),
         _input_buffer() {} 

    inline void init(size_t* problem_size)
    {
        _problem_size = *problem_size;
        _input = (float*)malloc(sizeof(float) * _problem_size);
        fill_random(_input, _input + _problem_size, -1.0f, 1.0f);

        _input_buffer.init(_context, mem_flag::read, _problem_size);
        _partial_buffer = opencl_buffer<float>();
    }

    inline void work_dimensions(size_t* group_dim)
    {
        _group_dim = group_dim[0];
        _group_num = (size_t)ceil((float)_problem_size / _group_dim);

        _partial = (float*)malloc(sizeof(float) * _group_num);
        _partial_buffer.init(_context, mem_flag::read_write, _group_num);

        _kernel->allocate_local_memory(2, _group_dim * sizeof(float));
    }

    inline void prepare()
    {
        _gpu = 0;
        fill(_partial, _partial + _group_num, 0.0f);

        _input_buffer.enqueue_write(_cmd_queue, _input);
        _partial_buffer.enqueue_write(_cmd_queue, _partial);
        clFinish(_cmd_queue);
    }

    inline void run_gpu()
    {
        _kernel->enqueue_args(&_input_buffer, &_partial_buffer);
        _kernel->enqueue_run(_cmd_queue, 1, &_problem_size, &_group_dim, NULL);

        _partial_buffer.enqueue_read(_cmd_queue, _partial);
        clFinish(_cmd_queue);  

        _gpu = kahan_reduce(_partial, _partial + _group_num, 0.0f);
    }

    inline void run_cpu()
    {
        _cpu = kahan_reduce(_input, _input + _problem_size, 0.0);
    }

    inline bool check_result()
    {
        return ::check_result(_gpu, _cpu);
    }

    inline void teardown()
    {
        return;
    }

    inline ~local_1d()
    {
        free(_partial);
        free(_input);
    }
};

struct global_2d
{
    size_t _problem_size[2];
    size_t _group_dim[2];
    size_t _group_num[2];
    float* _input;
    float* _partial;
    float _gpu;
    float _cpu;

    opencl_kernel* _kernel;
    cl_context _context;
    cl_command_queue _cmd_queue;

    opencl_buffer<float> _partial_buffer;
    opencl_buffer<float> _input_buffer;

    inline global_2d(cl_context context,
                     cl_command_queue cmd_queue,
                    opencl_kernel* kernel)
        :_problem_size{0, 0},
         _group_dim{0,0},
         _group_num{0,0},
         _input(NULL),
         _partial(NULL),
         _gpu(0),
         _cpu(0),
         _kernel(kernel),
         _context(context),
         _cmd_queue(cmd_queue),
         _partial_buffer(),
         _input_buffer() {} 

    inline void init(size_t* problem_size)
    {
        _problem_size[0] = problem_size[0];
        _problem_size[1] = problem_size[1];

        _input = (float*)malloc(sizeof(float) * _problem_size[0] * _problem_size[1]);
        fill_random(_input, _input + _problem_size[0], -1.0f, 1.0f);

        _input_buffer.init(_context, mem_flag::read, _problem_size[0] * _problem_size[1]);
        _partial_buffer = opencl_buffer<float>();

    }

    inline void work_dimensions(size_t* group_dim)
    {
        _group_dim[0] = group_dim[0];
        _group_dim[1] = group_dim[1];
        
        _group_num[0] = (size_t)ceil((float)_problem_size[0] / _group_dim[0]);
        _group_num[1] = (size_t)ceil((float)_problem_size[1] / _group_dim[1]);

        _partial = (float*)malloc(sizeof(float) * _group_num[0] * _group_num[1]);
        _partial_buffer.init(_context,
                             mem_flag::read_write,
                             _group_num[0] * _group_num[1]);
    }

    inline void prepare()
    {
        _gpu = 0;
        fill(_partial, _partial + (_group_num[0] * _group_num[1]), 0.0f);

        _input_buffer.enqueue_write(_cmd_queue, _input);
        _partial_buffer.enqueue_write(_cmd_queue, _partial);
        clFinish(_cmd_queue);
    }

    inline void run_gpu()
    {
        _kernel->enqueue_args(&_input_buffer, &_partial_buffer);
        _kernel->enqueue_run(_cmd_queue, 2, _problem_size, _group_dim, NULL);
        
        _partial_buffer.enqueue_read(_cmd_queue, _partial);
        clFinish(_cmd_queue);  

        _gpu = kahan_reduce(_partial, _partial + (_group_num[0] * _group_num[1]), 0.0f);
    }

    inline void run_cpu()
    {
        _cpu = kahan_reduce(_input, _input + (_problem_size[0] + _problem_size[1]), 0.0);
    }

    inline bool check_result()
    {
        return ::check_result(_gpu, _cpu);
    }

    inline void teardown()
    {
        return;
    }

    inline ~global_2d()
    {
        free(_partial);
        free(_input);
    }
};

#endif
