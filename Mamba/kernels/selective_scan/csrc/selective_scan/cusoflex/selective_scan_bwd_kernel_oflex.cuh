/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, bool kDeltaSoftplus_, typename input_t_, typename weight_t_, typename output_t_>
struct Selective_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;

    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int MaxDState = MAX_DSTATE;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && 3;
    static constexpr int kNLoadsOutput = sizeof(output_t) * kNLoads / kNBytes;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadOutputT = cub::BlockLoad<output_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadOutputVecT = cub::BlockLoad<vec_t, kNThreads, kNLoadsOutput, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockLoadOutputT::TempStorage),
                                                 sizeof(typename BlockLoadOutputVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = 2 * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using output_t = typename Ktraits::output_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load1 = reinterpret_cast<typename Ktraits::BlockLoadOutputT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 2 * Ktraits::MaxDState + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + Ktraits::MaxDState);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;

    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (batch_id * params.dB_batch_stride + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (batch_id * params.dC_batch_stride + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;
    float dD_val = 0;
    float ddelta_bias_val = 0;

    output_t *dout = reinterpret_cast<output_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride + dim_id * params.dout_d_stride;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize;
    Cvar += (params.n_chunks - 1) * kChunkSize;
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals[kNItems];
        input_t delta_vals_load[kNItems];
        float dout_vals[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        if constexpr (std::is_same_v<output_t, input_t>) {
            input_t dout_vals_load[kNItems];
            load_input<Ktraits>(reinterpret_cast<input_t *>(dout), dout_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            Converter<typename Ktraits::input_t, kNItems>::to_float(dout_vals_load, dout_vals);
        } else {
            static_assert(std::is_same_v<output_t, float>);
            load_output<Ktraits>(dout, dout_vals, smem_load1, params.seqlen - chunk * kChunkSize);
        }
        u -= kChunkSize;
        // Will reload delta at the same location if kDeltaSoftplus
        if constexpr (!kDeltaSoftplus) { delta -= kChunkSize; }
        dout -= kChunkSize;

        float delta_vals[kNItems];
        float du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { du_vals[i] = D_val * dout_vals[i]; }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { dD_val += dout_vals[i] * float(u_vals[i]); }

        float ddelta_vals[kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            constexpr float kLog2e = M_LOG2E;
            weight_t A_val = A[state_idx * params.A_dstate_stride];
            weight_t A_scaled = A_val * kLog2e;
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
            auto &smem_load_weight_C = smem_load_weight1;
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize));
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float delta_a_exp = exp2f(delta_vals[i] * A_scaled);
                thread_data[i] = make_float2(delta_a_exp, delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                if (i == 0) {
                    smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * Ktraits::MaxDState: threadIdx.x + 2 * Ktraits::MaxDState] = delta_a_exp;
                } else {
                    thread_reverse_data[i - 1].x = delta_a_exp;
                }
                thread_reverse_data[i].y = dout_vals[i] * C_vals[i];
            }
            __syncthreads();
            thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * Ktraits::MaxDState])
                    : smem_delta_a[threadIdx.x + 1 + 2 * Ktraits::MaxDState];
            // Initialize running total
            scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );
            scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
            Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
            );
            if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
            weight_t dA_val = 0;
            weight_t dB_vals[kNItems], dC_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float dx = thread_reverse_data[i].y;
                const float ddelta_u = dx * B_vals[i];
                du_vals[i] += ddelta_u * delta_vals[i];
                const float a = thread_data[i].y - (delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                ddelta_vals[i] += ddelta_u * float(u_vals[i]) + dx * A_val * a;
                dA_val += dx * delta_vals[i] * a;
                dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]);
                dC_vals[i] = dout_vals[i] * thread_data[i].y;
            }
            // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
            Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
            auto &smem_exchange_C = smem_exchange1;
            Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
            const int seqlen_remaining = params.seqlen - chunk * kChunkSize - threadIdx.x;
            weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
            weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i * kNThreads < seqlen_remaining) {
                    { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                    { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                }
            }
            dA_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
            if (threadIdx.x == 0) {
                smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
            }
        }

        if constexpr (kDeltaSoftplus) {
            input_t delta_vals_load[kNItems];
            __syncthreads();
            load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            delta -= kChunkSize;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float delta_val = float(delta_vals_load[i]) + delta_bias;
                float delta_val_neg_exp = expf(-delta_val);
                ddelta_vals[i] = delta_val <= 20.f
                    ? ddelta_vals[i] / (1.f + delta_val_neg_exp)
                        : ddelta_vals[i];
            }
        }
        
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { ddelta_bias_val += ddelta_vals[i]; }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id * params.du_d_stride + chunk * kChunkSize;
        input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
            + dim_id * params.ddelta_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        store_output<Ktraits>(ddelta, ddelta_vals, smem_store, params.seqlen - chunk * kChunkSize);
        Bvar -= kChunkSize;
        Cvar -= kChunkSize;
    }

    if (params.dD_ptr != nullptr) {
        __syncthreads();
        dD_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        __syncthreads();
        ddelta_bias_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
    }
    __syncthreads();
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
            using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, kDeltaSoftplus, input_t, weight_t, output_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * Ktraits::MaxDState) * sizeof(typename Ktraits::weight_t);
            // printf("smem_size = %d\n", kSmemSize);
            dim3 grid(params.batch, params.dim);
            auto kernel = &selective_scan_bwd_kernel<Ktraits>;
            if (kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<int knrows, typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_bwd_launch<32, 4, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_bwd_launch<32, 8, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_bwd_launch<32, 16, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_bwd_launch<64, 16, input_t, weight_t, output_t>(params, stream);
    } else {
        selective_scan_bwd_launch<128, 16, input_t, weight_t, output_t>(params, stream);
    }
}

