
#ifndef _OPENCL_KERNEL_H_
#define _OPENCL_KERNEL_H_

#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#endif

#include "OpenCL_util.h"

class opencl_kernel
{
private:
    cl_kernel _kernel;

    template<typename Type>
    inline cl_int arg_eval_impl(int num, Type arg)
    {
        return clSetKernelArg(_kernel, num, sizeof(cl_mem), arg);
    }

    template<typename Type, typename... Args>
    inline cl_int arg_eval_impl(int num, Type arg, Args... args)
    {
        cl_int err;
        err = clSetKernelArg(_kernel, num, sizeof(cl_mem), arg);

        if(err != CL_SUCCESS)
            return err;

        return arg_eval_impl(++num, args...);
    }

public:
    inline opencl_kernel(cl_kernel kernel)
        : _kernel(kernel)
    {
    }

    template<typename... Args>
    inline cl_int operator()(cl_command_queue queue,
                             cl_uint work_dim,
                             size_t const* global_work_size,
                             size_t const* local_work_size,
                             cl_event* event,
                             Args... args)
    {
        int num = 0;
        cl_int err;
        err = arg_eval_impl(num, args...);

        if(err != CL_SUCCESS)
            return err;

        return clEnqueueNDRangeKernel(
            queue, _kernel, work_dim, NULL,
            global_work_size, local_work_size, 0, NULL, event);
    }

    ~kernel()
    {
        clReleaseKernel(_kernel);
    }
};

template<size_t ProgNum>
class opencl_program
{
    cl_context* _context;
    cl_program _program;

public:
    opencl_program(cl_context* context,
                   cl_device_id* devices,
                   size_t device_num,
                   char const** progs,
                   size_t* lenghts,
                   size_t prog_num,
                   cl_int* err_code)
        : _context(context)
    {
        _program = clCreateProgramWithSource(*context,
                                             prog_num,
                                             progs,
                                             lenghts,
                                             err_code);   
        err_code = clBuildProgram(_program,
                                  device_num,
                                  devices,
                                  NULL,
                                  NULL,
                                  NULL);
    }

    opencl_kernel create_kernel(char const* name, cl_int* err_code)
    {
        cl_kernel kernel = clCreateKernel(_program, name, err_code);
        return opencl_kernel(kernel);
    }

    ~opencl_program()
    {
        clReleaseProgram(_program);
    }
};


#endif
