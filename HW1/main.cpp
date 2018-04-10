
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "OpenCL_util.h"
#include "personal_utils.h"
#include "opencl_wrapper.h"
#include "benchmark_suites.h"

void global_1d_benchmark(cl_context* context,
                         cl_command_queue* queue,
                         opencl_kernel* kernel)
{
    printf("-- parallel reduction 1d global memory\n");
    global_1d bench(*context, *queue, kernel);            

    size_t group_size = 64;
    for(size_t i = 0; i < 11; ++i)
    {
        size_t problem_size = 1024 << i;
        run_benchmark(std::move(bench), &group_size, &problem_size, 1);
    }
    printf("\n\n");
}

void local_1d_benchmark(cl_context* context,
                        cl_command_queue* queue,
                        opencl_kernel* kernel)
{
    printf("-- parallel reduction 1d local memory\n");
    local_1d bench(*context, *queue, kernel);            

    size_t group_size = 64;
    for(size_t i = 0; i < 11; ++i)
    {
        size_t problem_size = 1024 << i;
        run_benchmark(std::move(bench), &group_size, &problem_size, 1);
    }
    printf("\n\n");
}

void global_2d_benchmark(cl_context* context,
                         cl_command_queue* queue,
                         opencl_kernel* kernel)
{
    printf("-- parallel reduction 2d global memory\n");
    global_2d bench(*context, *queue, kernel);            

    size_t group_size[2] = {16, 16};
    for(size_t i = 0; i < 7; ++i)
    {
        size_t root = 16 << i;
        size_t problem_size[2] = {root, root};
        run_benchmark(std::move(bench), group_size, problem_size, 2);
    }
    printf("\n\n");
}

void local_2d_benchmark(cl_context* context,
                        cl_command_queue* queue,
                        opencl_kernel* kernel)
{
    printf("-- parallel reduction 2d local memory\n");
    local_2d bench(*context, *queue, kernel);            

    size_t group_size[2] = {16, 16};
    for(size_t i = 0; i < 7; ++i)
    {
        size_t root = 16 << i;
        size_t problem_size[2] = {root, root};
        run_benchmark(std::move(bench), group_size, problem_size, 2);
    }
    printf("\n\n");
}

void group_size_benchmark(cl_context* context,
                          cl_command_queue* queue,
                        opencl_kernel* kernel)
{
    printf("-- parallel reduction 1d local memory by group size\n");
    local_1d bench(*context, *queue, kernel);            

    for(size_t group_size = 64; group_size <= 1024; group_size <<= 1)
    {
        printf("[ group size: %d ]\n", (int)group_size);

        for(size_t problem_size = 1024;
            problem_size <= 1024 * 1024;
            problem_size <<= 1)
        {
            run_benchmark(std::move(bench), &group_size, &problem_size, 1);
        }
    }
    printf("\n\n");
}

void double_reduce_benchmark(cl_context* context,
                             cl_command_queue* queue,
                             opencl_kernel* kernel)
{
    printf("-- parallel reduction 1d local memory\n");
    double_reduce bench(*context, *queue, kernel);            

    size_t group_size = 64;
    for(size_t i = 3; i < 11; ++i)
    {
        size_t problem_size = 1024 << i;
        run_benchmark(std::move(bench), &group_size, &problem_size, 1);
    }
    printf("\n\n");
}

int main(void) {
    if (false) {
        show_OpenCL_platform();
        return 0;
    }
    
    cl_int errcode_ret;
    cl_platform_id platform;
    cl_device_id device;
    
    CHECK_ERROR_CODE(
        clGetPlatformIDs(1, &platform, NULL));  // You may skip error checking if you think it is unnecessary.
    
    CHECK_ERROR_CODE(
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    print_device_0(device);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);

    char* kernel_sources;
    size_t size = read_file("./reduce_sum.cl", &kernel_sources, false);

    opencl_program prog(&context, &device, 1, kernel_sources, size, 1);
    opencl_kernel reduce1 = prog.create_kernel("reduce");
    opencl_kernel reduce2 = prog.create_kernel("reduce_local");
    opencl_kernel reduce3 = prog.create_kernel("reduce_2d");
    opencl_kernel reduce4 = prog.create_kernel("reduce_2d_local");

    //global_1d_benchmark(&context, &cmd_queue, &reduce1);
    //local_1d_benchmark(&context, &cmd_queue, &reduce2);
    //global_2d_benchmark(&context, &cmd_queue, &reduce3);
    //local_2d_benchmark(&context, &cmd_queue, &reduce4);
    //group_size_benchmark(&context, &cmd_queue, &reduce2);
    double_reduce_benchmark(&context, &cmd_queue, &reduce2);

    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
    
    return 0;
}
