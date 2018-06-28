//
//  simple_kernel2.cl
//  Simple_SIMT
//
//  Written for CSEG437/CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2018년 Sogang University. All rights reserved.
//

__kernel void CombineTwoArrays2( __global float* A, __global float* B, __global float* C  ) {
    int num_groups = get_num_groups(0);
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    
    int i = local_id * num_groups + group_id;
    
    C[i] = 1.0f / (sin(A[i])*cos(B[i]) + cos(A[i])*sin(B[i]));
}
