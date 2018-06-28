//
//  simple_kernel.cl
//  Simple_SIMT
//
//  Written for CSEG437/CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2018년 Sogang University. All rights reserved.
//

// By uncommenting the qualifier "__attribute__((reqd_work_group_size(128, 1, 1)))", you may specify
// the work-group size that must be used as the local_work_size argument to clEnqueueNDRangeKernel.
// This allows the compiler to optimize the generated code appropriately for this kernel.

__kernel
//__attribute__((reqd_work_group_size(128, 1, 1)))
void CombineTwoArrays( __global float* A, __global float* B, __global float* C ) {
    int i = get_global_id(0);

    C[i] = 1.0f / (sin(A[i])*cos(B[i]) + cos(A[i])*sin(B[i]));
}
