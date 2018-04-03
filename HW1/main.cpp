//
//  main.cpp
//  Simple_SIMT
//
//  Written for CSEG437/CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2018년 Sogang University. All rights reserved.
//

#include<math.h>
#include<time.h>
#include<stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#endif

#include "OpenCL_util.h"

//#define COALESCED_GLOBAL_MEMORY_ACCESS  // What happens if this line is commented out?

#ifdef COALESCED_GLOBAL_MEMORY_ACCESS
#define OPENCL_C_PROG_FILE_NAME "simple_kernel.cl"
#define KERNEL_NAME "CombineTwoArrays"
#else
#define OPENCL_C_PROG_FILE_NAME "simple_kernel2.cl"
#define KERNEL_NAME "CombineTwoArrays2"
#endif

//////////////////////////////////////////////////////////////////////////
void generate_random_float_array(float *array, int n) {
    srand((unsigned int)201803); // Always the same input data
    for (int i = 0; i < n; i++) {
        array[i] = 3.1415926f*((float)rand() / RAND_MAX);
    }
}

void combine_two_arrays_CPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = 1.0f / (sin(A[i])*cos(B[i]) + cos(A[i])*sin(B[i]));
    }
}
//////////////////////////////////////////////////////////////////////////

#define INDEX_GPU 0
#define INDEX_CPU 1

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char *string;
} OPENCL_C_PROG_SRC;

int main(void) {
    cl_int errcode_ret;
    float compute_time;
    
    size_t n_elements, work_group_size_GPU, work_group_size_CPU;
    float *array_A, *array_B, *array_C;
    OPENCL_C_PROG_SRC prog_src;
    
    cl_platform_id platform;
    cl_device_id devices[2];
    cl_context context;
    cl_command_queue cmd_queues[2];
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_A, buffer_B, buffer_C_GPU, buffer_C_CPU;
    cl_event event_for_timing;
    
    if (0) {
        // Just to reveal my OpenCl platform...
        show_OpenCL_platform();
        return 0;
    }
    
    n_elements = 128 * 1024 * 1024;
    work_group_size_GPU = 4; // What would happen if it is 2, 4, 8, 16, 32, 64, 128, 512 or 1024?
    work_group_size_CPU = 4; // What would happen if it is 2, 4, 8, 16, 32, 64, 128, 512 or 1024?
    
    array_A = (float *)malloc(sizeof(float)*n_elements);
    array_B = (float *)malloc(sizeof(float)*n_elements);
    array_C = (float *)malloc(sizeof(float)*n_elements);
    
    fprintf(stdout, "^^^ Generating random input arrays with %d elements each...\n", (int) n_elements);
    generate_random_float_array(array_A, (int) n_elements);
    generate_random_float_array(array_B, (int) n_elements);
    fprintf(stdout, "^^^ Done!\n");
    
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    fprintf(stdout, "\n^^^ Test 1: general CPU computation ^^^\n");
    fprintf(stdout, "   [CPU Execution] \n");
    CHECK_TIME_START;
    combine_two_arrays_CPU(array_A, array_B, array_C, (int) n_elements);
    CHECK_TIME_END(compute_time);
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    fprintf(stdout, "   [Check Results] \n");
    fprintf(stdout, "     * C[%lu] = %f, C[%lu] = %f, C[%lu] = %f\n\n", n_elements/3, array_C[n_elements / 3],
            n_elements / 2, array_C[n_elements / 2], 3 * n_elements / 4, array_C[3 * n_elements / 4]);
    
    memset(array_C, 0, sizeof(float)*n_elements);
    
    /* Get the first platform. */
    // You may obtain the list of platforms available if you want.
    errcode_ret = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR_CODE(errcode_ret);  // You may skip error checking if you think it is unnecessary.
    
    /* Get the first GPU device. */
    // You may obtain the list of devices available on a platform. clGetDeviceIDs may return all or
    // a subset of the actual physical devices present in the platform and that match device_type.
    errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &devices[INDEX_GPU], NULL);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
    print_device_0(devices[INDEX_GPU]);
    
    /* Get the first CPU device. */
    errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &devices[INDEX_CPU], NULL);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "\n^^^ The first CPU device on the platform ^^^\n");
    print_device_0(devices[INDEX_CPU]);
    
    /* Create a context with the devices. */
    // An OpenCL context is created with one or more devices. Contexts are used by the OpenCL runtime
    // for managing objects such as command queues, memory, program and kernel objects and
    // for executing kernels on one or more devices specified in the context.
    context = clCreateContext(NULL, 2, devices, NULL, NULL, &errcode_ret);
    
    /* Create a command-queue for the GPU device. */
    // Create a command-queue on a specific device. OpenCL objects such as memory, program and
    // kernel objects are created using a context. Operations on these objects are performed using
    // a command-queue. The command-queue can be used to queue a set of operations (referred to as commands)
    // in order. Having multiple command-queues allows applications to queue multiple independent commands
    // without requiring synchronization. Note that this should work as long as these objects
    // are not being shared. Sharing of objects across multiple command-queues will require the application
    // to perform appropriate synchronization.
    // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    cmd_queues[INDEX_GPU] = clCreateCommandQueue(context, devices[INDEX_GPU], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    
    /* Create a command queue for the CPU device. */
    cmd_queues[INDEX_CPU] = clCreateCommandQueue(context, devices[INDEX_CPU], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    
    /* Create a program from OpenCL C source code. */
    prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, &prog_src.string);
    // Creates a program object for a context, and loads the source code specified by the text strings
    // in the strings array into the program object. This function creates a program object for a context,
    // and loads the source code specified by the text strings in the strings array into the program object.
    // The devices associated with the program object are the devices associated with context.
    // The source code specified by strings is either an OpenCL C program source, header or
    // implementation-defined source for custom devices that support an online compiler.
    program = clCreateProgramWithSource(context, 1, (const char **) &prog_src.string, &prog_src.length, &errcode_ret);
    
    /* Build a program executable from the program object. */
    // Builds (compiles and links) a program executable from the program source or binary for all the devices
    // or a specific device(s) in the OpenCL context associated with program.
    // OpenCL allows program executables to be built using the source or the binary.
    // clBuildProgram must be called for program created using either clCreateProgramWithSource or
    // clCreateProgramWithBinary to build the program executable for one or more devices
    // associated with program. If program is created with clCreateProgramWithBinary, then the program
    // binary must be an executable binary(not a compiled binary or library). The executable binary can be
    // queried using clGetProgramInfo(program, CL_PROGRAM_BINARIES, ...) and can be specified to
    // clCreateProgramWithBinary to create a new program object.
    errcode_ret = clBuildProgram(program, 2, devices, NULL, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(program, devices[INDEX_GPU], "GPU");
        print_build_log(program, devices[INDEX_CPU], "CPU");
        exit(-1);
    }
    
    /* Create the kernel from the program. */
    // Creates a kernel object. A kernel is a function declared in a program.
    // A kernel is identified by the __kernel qualifier applied to any function in a program.
    // A kernel object encapsulates the specific __kernel function declared in a program and
    // the argument values to be used when executing this __kernel function.
    kernel = clCreateKernel(program, KERNEL_NAME, &errcode_ret);
    
    /* Create input and output buffer objects. */
    // The user is responsible for ensuring that data passed into and out of OpenCL images
    // are natively aligned relative to the start of the buffer as per kernel language or IL requirements.
    // OpenCL buffers created with CL_MEM_USE_HOST_PTR need to provide an appropriately aligned host memory
    // pointer that is aligned to the data types used to access these buffers in a kernel(s).
    buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);
    
    buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);
    
    buffer_C_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);
    
    buffer_C_CPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "   [Data Transfer to GPU] \n");
    
    CHECK_TIME_START;
    // Move the input data from the host memory to the GPU device memory.
    errcode_ret = clEnqueueWriteBuffer(cmd_queues[INDEX_GPU], buffer_A, CL_FALSE, 0,
                                       sizeof(float)*n_elements, array_A, 0, NULL, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    
    errcode_ret = clEnqueueWriteBuffer(cmd_queues[INDEX_GPU], buffer_B, CL_FALSE, 0,
                                       sizeof(float)*n_elements, array_B, 0, NULL, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    
    /* Wait until all data transfers finish. */
    // Blocks until all previously queued OpenCL commands in a command-queue are issued to
    // the associated device and have completed. clFinish does not return until all previously
    // queued commands in command_queue have been processed and completed. clFinish is also
    // a synchronization point.
    clFinish(cmd_queues[INDEX_GPU]); // What if this line is removed?
    CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    
    /* Set the kenel arguments. */
    // Set the argument value for a specific argument of a kernel.
    errcode_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    CHECK_ERROR_CODE(errcode_ret);
    
    errcode_ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    CHECK_ERROR_CODE(errcode_ret);
    
    errcode_ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C_GPU);
    CHECK_ERROR_CODE(errcode_ret);
    
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    fprintf(stdout, "\n^^^ Test 2: Computing on OpenCL GPU Device ^^^\n");
    
    printf_KernelWorkGroupInfo(kernel, devices[INDEX_GPU]);
    
    fprintf(stdout, "   [Kernel Execution] \n");
    
    /* Execute the kernel on the device. */
    // Enqueues a command to execute a kernel on a device.
    CHECK_TIME_START;
    errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_GPU], kernel, 1, NULL,
                                         &n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
    CHECK_ERROR_CODE(errcode_ret);
    clFinish(cmd_queues[INDEX_GPU]);  // What would happen if this line is removed?
    // or clWaitForEvents(1, &event_for_timing);
    CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
    
    fprintf(stdout, "   [Data Transfer] \n");
    
    /* Read back the device buffer to the host array. */
    // Enqueue commands to read from a buffer object to host memory.
    CHECK_TIME_START;
    errcode_ret =  clEnqueueReadBuffer(cmd_queues[INDEX_GPU], buffer_C_GPU, CL_TRUE, 0,
                                       sizeof(float)*n_elements, array_C, 0, NULL, &event_for_timing);
    CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    // In this case, you do not need to call clFinish() for a synchronization.
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
    
    fprintf(stdout, "   [Check Results] \n");
    fprintf(stdout, "    * C[%lu] = %f, C[%lu] = %f, C[%lu] = %f\n\n", n_elements / 3, array_C[n_elements / 3],
            n_elements / 2, array_C[n_elements / 2], 3 * n_elements / 4, array_C[3 * n_elements / 4]);
    
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    fprintf(stdout, "\n^^^ Test 3: Computing on OpenCL CPU Device ^^^\n");
    
    memset(array_C, 0, sizeof(float)*n_elements); // Erase the GPU results.
    
    /* Set the argument again only when it is necessary. */
    errcode_ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C_CPU);
    CHECK_ERROR_CODE(errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);
    
    printf_KernelWorkGroupInfo(kernel, devices[INDEX_CPU]);
    
    // Where is the codes that move  the input data from the host memory to the CPU device memory?
    
    fprintf(stdout, "   [Kernel Execution] \n");
    
    CHECK_TIME_START;
    errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_CPU], kernel, 1, NULL,
                                         &n_elements, &work_group_size_CPU, 0, NULL, &event_for_timing);
    CHECK_ERROR_CODE(errcode_ret);
    clFinish(cmd_queues[INDEX_CPU]);
    CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
    
    fprintf(stdout, "   [Data Transfer] \n");
    
    CHECK_TIME_START;
    errcode_ret = clEnqueueReadBuffer(cmd_queues[INDEX_CPU], buffer_C_CPU, CL_TRUE, 0,
                                      sizeof(float)*n_elements, array_C, 0, NULL, &event_for_timing);
    CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
    
    fprintf(stdout, "   [Check Results] \n");
    fprintf(stdout, "    * C[%ld] = %f, C[%ld] = %f, C[%ld] = %f\n\n", n_elements / 3, array_C[n_elements / 3],
            n_elements / 2, array_C[n_elements / 2], 3 * n_elements / 4, array_C[3 * n_elements / 4]);
    
    /* Free OpenCL resources. */
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C_GPU);
    clReleaseMemObject(buffer_C_CPU);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmd_queues[INDEX_GPU]);
    clReleaseCommandQueue(cmd_queues[INDEX_CPU]);
    clReleaseContext(context);
    
    /* Free host resources. */
    free(array_A);
    free(array_B);
    free(array_C);
    free(prog_src.string);
    
    return 0;
}


