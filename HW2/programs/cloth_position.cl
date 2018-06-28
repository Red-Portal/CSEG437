__kernel
void cloth_position(
    __global float4* pos_in, __global float4* pos_out,
    __global float4* vel_in, __global float4* vel_out,
    __local float4* local_data,
    float3 Gravity,
    float ParticleMass,
    float ParticleInvMass,
    float SpringK,
    float RestLengthHoriz,
    float RestLengthVert,
    float RestLengthDiag,
    float DeltaT,
    float DampingConst) {
    int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

}