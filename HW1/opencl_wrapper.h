
#ifndef _OPENCL_WRAPPER_H_
#define _OPENCL_WRAPPER_H_

#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "OpenCL_util.h"

enum class mem_flag
{
    read = CL_MEM_READ_ONLY,
    write = CL_MEM_WRITE_ONLY,
    read_write = CL_MEM_READ_WRITE
};

template<typename Type>
struct opencl_buffer
{
    size_t _size;
    cl_mem _buffer;

    inline opencl_buffer() = default;

    inline void init(cl_context context, mem_flag flag, size_t n)
    {
        _size = n;
        cl_int errcode_ret;
        _buffer = clCreateBuffer(context,
                                 (cl_mem_flags)flag,
                                 sizeof(Type) * n,
                                 NULL,
                                 &errcode_ret);
        CHECK_ERROR_CODE(errcode_ret);
    }

    inline opencl_buffer(cl_context context, mem_flag flag, size_t n)
        : _size(n)

    {
        cl_int errcode_ret;
        _buffer = clCreateBuffer(context,
                                 (cl_mem_flags)flag,
                                 sizeof(Type) * n,
                                 NULL,
                                 &errcode_ret);
        CHECK_ERROR_CODE(errcode_ret);
    }

    inline void enqueue_read(cl_command_queue queue, Type* ptr)
    {
        CHECK_ERROR_CODE(
            clEnqueueReadBuffer(queue, _buffer, CL_FALSE, 0,
                                sizeof(Type)*_size, (void*)ptr, 0, NULL, NULL));
    }

    inline void enqueue_write(cl_command_queue queue, Type const* ptr)
    {
        CHECK_ERROR_CODE(
            clEnqueueWriteBuffer(queue, _buffer, CL_FALSE, 0,
                                 sizeof(Type)*_size, (void const*)ptr, 0, NULL, NULL));
    }

    inline ~opencl_buffer()
    {
        clReleaseMemObject(_buffer);
    }
};

struct opencl_kernel
{
    cl_kernel _kernel;

    template<typename Type>
    inline void arg_eval_impl(int num, opencl_buffer<Type>* arg)
    {
        CHECK_ERROR_CODE(
            clSetKernelArg(_kernel, num, sizeof(cl_mem), &arg->_buffer));
    }

    template<typename Type, typename... Args>
    inline void arg_eval_impl(int num,
                              opencl_buffer<Type>* arg,
                              Args... args)
    {
        CHECK_ERROR_CODE(
            clSetKernelArg(_kernel, num, sizeof(cl_mem), &arg->_buffer));
        arg_eval_impl(++num, args...);
    }

    inline explicit opencl_kernel(cl_kernel kernel)
        : _kernel(kernel) {}

    template<typename... Args>
    inline void enqueue_args(Args... args)
    {
        int num = 0;
        arg_eval_impl(num, args...);
    }

    inline void allocate_local_memory(size_t arg_num, size_t memory_size)
    {
        CHECK_ERROR_CODE(
            clSetKernelArg(_kernel, arg_num, memory_size, NULL));
    }

    inline void enqueue_run(cl_command_queue queue,
                            cl_uint work_dim,
                            size_t const* global_work_size,
                            size_t const* local_work_size,
                            cl_event* event)
    {
        CHECK_ERROR_CODE(
            clEnqueueNDRangeKernel(
                queue, _kernel, work_dim, NULL,
                global_work_size, local_work_size, 0, NULL, event));
    }

    ~opencl_kernel()
    {
        clReleaseKernel(_kernel);
    }
};

struct opencl_program
{
    cl_context* _context;
    cl_program _program;

    opencl_program(cl_context* context,
                   cl_device_id* devices,
                   size_t device_num,
                   char const* progs,
                   size_t length,
                   size_t prog_num)
        : _context(context)
    {
        cl_int err_code;
        _program = clCreateProgramWithSource(*context,
                                             prog_num,
                                             &progs,
                                             &length,
                                             &err_code);   
        CHECK_ERROR_CODE(err_code);

        if(clBuildProgram(_program,
                          device_num,
                          devices,
                          NULL,
                          NULL,
                          NULL) != CL_SUCCESS)
        {
            size_t log_len;
            CHECK_ERROR_CODE(
                clGetProgramBuildInfo(_program,
                                      *devices,
                                      CL_PROGRAM_BUILD_LOG,
                                      0,
                                      NULL,
                                      &log_len));
            

            char* error_buff = (char*)malloc(log_len);
            if (!error_buff) {
                printf("malloc failed at line %d\n", __LINE__);
                exit(-2);
            }

            CHECK_ERROR_CODE(
                clGetProgramBuildInfo(_program,
                                      *devices,
                                      CL_PROGRAM_BUILD_LOG,
                                      log_len,
                                      error_buff,
                                      NULL));

            fprintf(stderr,"Build log: \n%s\n", error_buff); //Be careful with  the fprint
            free(error_buff);
            fprintf(stderr,"clBuildProgram failed\n");
            exit(EXIT_FAILURE);
        }

    }

    opencl_kernel create_kernel(char const* name)
    {
       cl_int err_code;
       cl_kernel kernel = clCreateKernel(_program, name, &err_code);
       CHECK_ERROR_CODE(err_code);

       return opencl_kernel(kernel);
    }

    ~opencl_program()
    {
        clReleaseProgram(_program);
    }
};


#endif
