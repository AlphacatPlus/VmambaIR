/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_fwd_kernel_nrow.cuh"

template void selective_scan_fwd_cuda<2, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<2, at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<2, at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);

