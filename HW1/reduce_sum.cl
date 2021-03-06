
__kernel void reduce(__global float* vec, 
                     __global float* sum)
{
    int idx = get_global_id (0); 
    int local_idx = get_local_id (0); 
    int group_dim = get_local_size(0);
    int group_idx = get_group_id(0);

    for(unsigned int stride = group_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_idx < stride)
            vec[idx] += vec[idx + stride];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if(local_idx == 0u)
        sum[group_idx] = vec[idx];
}

__kernel void reduce_local(__global float* vec,
                           __global float* sum,
                           __local float* shared)
{
    int idx = get_global_id (0); 
    int local_idx = get_local_id (0); 
    int group_dim = get_local_size(0);
    int group_idx = get_group_id(0);

    shared[local_idx] = vec[idx];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int stride = group_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_idx < stride)
            shared[local_idx] += shared[local_idx + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_idx == 0u)
        sum[group_idx] = shared[local_idx];
}


__kernel void reduce_2d(__global float* vec,
                        __global float* sum)
{
    int group_col_idx = get_group_id(0);
    int group_row_idx = get_group_id(1);

    int group_col_dim = get_num_groups(0);
    int group_row_dim = get_num_groups(1);

    int local_col_idx = get_local_id(0);
    int local_row_idx = get_local_id(1);

    int local_col_dim = get_local_size(0);
    int local_row_dim = get_local_size(1);

    int col_idx = get_global_id(0);
    int row_idx = get_global_id(1);

    int col_dim = get_global_size(0);
    int row_dim = get_global_size(1);

    int idx = row_idx * col_dim + col_idx;

    for(unsigned int stride = local_col_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_col_idx < stride)
            vec[idx] += vec[idx + stride];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for(unsigned int stride = local_row_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_row_idx < stride)
            vec[idx] += vec[idx + (stride * col_dim)];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if((local_row_idx + local_col_idx) == 0)
        sum[group_row_idx * group_col_dim + group_col_idx] = vec[idx];
}


__kernel void reduce_2d_local(__global float* vec,
                              __global float* sum,
                              __local float* shared)
{
    int group_col_idx = get_group_id(0);
    int group_row_idx = get_group_id(1);

    int group_col_dim = get_num_groups(0);
    int group_row_dim = get_num_groups(1);

    int local_col_idx = get_local_id(0);
    int local_row_idx = get_local_id(1);

    int local_col_dim = get_local_size(0);
    int local_row_dim = get_local_size(1);

    int col_idx = get_global_id(0);
    int row_idx = get_global_id(1);

    int col_dim = get_global_size(0);
    int row_dim = get_global_size(1);

    int idx = row_idx * col_dim + col_idx;
    int local_idx = local_row_idx * local_col_dim + local_col_idx;
    
    shared[local_idx] = vec[idx];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int stride = local_col_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_col_idx < stride)
            shared[local_idx] += shared[local_idx + stride];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(unsigned int stride = local_row_dim / 2; stride > 0u; stride >>= 1)
    {
        if(local_row_idx < stride)
            shared[local_idx] += shared[local_idx + (stride * local_col_dim)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if((local_row_idx + local_col_idx) == 0)
        sum[group_row_idx * group_col_dim + group_col_idx] = shared[local_idx];
}
