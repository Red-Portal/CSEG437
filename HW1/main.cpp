
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

const char *source =
    "__kernel void add(__global float *a, __global float *b, __global float*c) { \n"
    "    /* Get unique id of work-item */ \n"
    "    int idx = get_global_id (0); \n"
    "    printf('%d ', idx); \n"
    "    /* make addition for `a` and `b`, store result into `c` */ \n"
    "    c[idx] = a[idx] + b[idx]; \n"
    "} \n";

template<typename Type>
Type reduce_vector(Type* v, size_t n)
{
    for (size_t i = n; i >= 1u; i >>= 1) {
        if(i % 2 != 0)
        {
            v[i - 2] += v[i - 1];
            --i;
        }

        size_t m = i / 2;

        for(size_t j = 0; j < m; j++) {
            v[j] += v[j + m];
        }
    }
    return v[0];
}

bool reduce_vector_batch(cl_context context,
                         cl_command_queue cmd_queues,
                         opencl_kernel kernel,
                         size_t problem_size,
                         size_t group_size)
{
    float* array;
    array = (float *)malloc(sizeof(float) * problem_size);
    fill_random(array, array + problem_size, -1.0, 1.0);

    cl_event event_for_timing;
    

    free(array);
}

int main(void) {
    cl_int errcode_ret;
    
    size_t n_elements, work_group_size;
    float *array_A, *array_B, *array_C;
    
    cl_context context;
    cl_platform_id platform;
    cl_device_id device;
    cl_command_queue cmd_queues;

    cl_event event_for_timing;
    
    if (false) {
        // Just to reveal my OpenCl platform...
        show_OpenCL_platform();
        return 0;
    }
    
    n_elements = 128 * 1024;
    work_group_size = 4; // What would happen if it is 2, 4, 8, 16, 32, 64, 128, 512 or 1024?

    
    array_B = (float *)malloc(sizeof(float)*n_elements);
    array_C = (float *)malloc(sizeof(float)*n_elements);
    
    fprintf(stdout, "^^^ Generating random input arrays with %d elements each...\n", (int) n_elements);
    fill_random(array_B, array_B + n_elements, -1.0, 1.0);
    fprintf(stdout, "^^^ Done!\n");

    printf("current line: %d", __LINE__);
    memset(array_C, 0, sizeof(float)*n_elements);

    CHECK_ERROR_CODE(
        clGetPlatformIDs(1, &platform, NULL));  // You may skip error checking if you think it is unnecessary.
    
    CHECK_ERROR_CODE(
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    print_device_0(device);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    opencl_buffer<float> buffer_A(context, mem_flag::read, n_elements);
    opencl_buffer<float> buffer_B(context, mem_flag::read, n_elements);
    opencl_buffer<float> buffer_C(context, mem_flag::write, n_elements);

    cmd_queues = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);

    size_t kernel_num = strlen(source);
    opencl_program<1> prog(&context, &device, 1, &source, &kernel_num, 1);
    opencl_kernel reduce = prog.create_kernel("add");

    fprintf(stdout, "   [Data Transfer to GPU] \n");
    
    benchmark(
        [&]()
        {
            buffer_A.enqueue_write(cmd_queues, array_A);
            buffer_B.enqueue_write(cmd_queues, array_B);
            clFinish(cmd_queues);
        });
    
    printf_KernelWorkGroupInfo(reduce._kernel, device);
    
    fprintf(stdout, "   [Kernel Execution] \n");
    benchmark(
        [&]()
        {
            reduce.enqueue_args(&buffer_A, &buffer_B, &buffer_C);
            reduce.enqueue_run(cmd_queues, 1, &n_elements, &work_group_size, &event_for_timing);
            clFinish(cmd_queues);  // What would happen if this line is removed?
        });

    print_device_time(event_for_timing);

    fprintf(stdout, "   [Data Transfer] \n");
    benchmark(
        [&]()
        {
            buffer_C.enqueue_read(cmd_queues, array_C);
            clFinish(cmd_queues);
        });


    fprintf(stdout, "   [Check Results] \n");
    if(check_result(array_A, array_B, array_C, n_elements))
        printf(" correct! \n");
    else
        printf(" wrong.. \n");
    
    /* Free OpenCL resources. */
    clReleaseCommandQueue(cmd_queues);
    clReleaseContext(context);
    
    /* Free host resources. */
    free(array_A);
    free(array_B);
    free(array_C);
    
    return 0;
}


