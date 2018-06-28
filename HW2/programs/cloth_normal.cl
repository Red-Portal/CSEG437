__kernel
void cloth_normal(
    __global float4* pos_in, __global float4* nor_out,
    __local float4* local_data) {
    int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

    // Copy into local memory
    uint local_width = get_local_size(0) + 2;

    local_data[(get_local_id(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx];

    // Bottom edge
    if (get_local_id(1) == 0)
    {
        if (get_global_id(1) > 0)
        {
            local_data[get_local_id(0) + 1] = pos_in[idx - get_global_size(0)];

            // Lower left corner
            if (get_local_id(0) == 0)
            {
                if (get_global_id(0) > 0)
                {
                    local_data[0] = pos_in[idx - get_global_size(0) - 1];
                }
                else
                {
                    local_data[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }

            // Lower right corner
            if (get_local_id(0) == get_local_size(0) - 1)
            {
                if (get_global_id(0) < get_global_size(0) - 1)
                {
                    local_data[get_local_size(0) + 1] = pos_in[idx - get_global_size(0) + 1];
                }
                else
                {
                    local_data[get_local_size(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        else
        {
            local_data[get_local_id(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Top edge
    if (get_local_id(1) == get_local_size(1) - 1)
    {
        if (get_global_id(1) < get_global_size(1) - 1)
        {
            local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx + get_global_size(0)];

            // Upper left corner
            if (get_local_id(0) == 0)
            {
                if (get_global_id(0) > 0)
                {
                    local_data[(get_local_size(1) + 1) * local_width] = pos_in[idx + get_global_size(0) - 1];
                }
                else
                {
                    local_data[(get_local_size(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }

            //Lower right corner
            if (get_local_id(0) == get_local_size(0) - 1)
            {
                if (get_global_id(0) < get_global_size(0) - 1)
                {
                    local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + get_global_size(0) + 1];
                }
                else
                {
                    local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        else
        {
            local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Left edge
    if (get_local_id(0) == 0)
    {
        if (get_global_id(0) > 0)
        {
            local_data[(get_local_id(1) + 1) * local_width] = pos_in[idx - 1];
        }
        else
        {
            local_data[(get_local_id(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Right edge
    if (get_local_id(0) == get_local_size(0) - 1)
    {
        if (get_global_id(0) < get_global_size(0) - 1)
        {
            local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + 1];
        }
        else
        {
            local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    uint idx_local = (get_local_id(1) + 1) * local_width + (get_local_id(0) + 1);

    float3 p = local_data[idx_local].xyz;
    float3 n = (float3)(0.0f, 0.0f, 0.0f);
    float3 a, b, c;

    if (get_global_id(1) < get_global_size(1) - 1) {
        c = local_data[idx_local + local_width].xyz - p;
        if (get_global_id(0) < get_global_size(0) - 1) {
            a = local_data[idx_local + 1].xyz - p;
            b = local_data[idx_local + local_width + 1].xyz - p;
            n += cross(a, b);
            n += cross(b, c);
        }
        if (get_global_id(0) > 0) {
            a = c;
            b = local_data[idx_local + local_width - 1].xyz - p;
            c = local_data[idx_local - 1].xyz - p;
            n += cross(a, b);
            n += cross(b, c);
        }
    }

    if (get_global_id(1) > 0) {
        c = local_data[idx_local - local_width].xyz - p;
        if (get_global_id(0) > 0) {
            a = local_data[idx_local - 1].xyz - p;
            b = local_data[idx_local - local_width - 1].xyz - p;
            n += cross(a, b);
            n += cross(b, c);
        }
        if (get_global_id(0) < get_global_size(0) - 1) {
            a = c;
            b = local_data[idx_local - local_width + 1].xyz - p;
            c = local_data[idx_local + 1].xyz - p;
            n += cross(a, b);
            n += cross(b, c);
        }
    }

    nor_out[idx] = (float4)(normalize(n), 0.0f);
}
