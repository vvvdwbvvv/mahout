//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "kernel_config.h"

namespace {

constexpr unsigned int ZZ_ENT_FULL = 0;
constexpr unsigned int ZZ_ENT_LINEAR = 1;
constexpr unsigned int ZZ_ENT_CIRCULAR = 2;

constexpr unsigned int ZZ_MAP_QISKIT = 0;
constexpr unsigned int ZZ_MAP_PRODUCT = 1;

__device__ __forceinline__ bool pair_in_entanglement(
    unsigned int i,
    unsigned int j,
    unsigned int num_qubits,
    unsigned int entanglement_mode
) {
    if (i >= j || j >= num_qubits) {
        return false;
    }

    switch (entanglement_mode) {
        case ZZ_ENT_FULL:
            return true;
        case ZZ_ENT_LINEAR:
            return j == i + 1;
        case ZZ_ENT_CIRCULAR:
            if (num_qubits <= 2) {
                return j == i + 1;
            }
            return (j == i + 1) || (i == 0 && j == num_qubits - 1);
        default:
            return false;
    }
}

__device__ __forceinline__ double pair_map_value(
    double xi,
    double xj,
    unsigned int map_func
) {
    switch (map_func) {
        case ZZ_MAP_QISKIT:
            return (M_PI - xi) * (M_PI - xj);
        case ZZ_MAP_PRODUCT:
            return xi * xj;
        default:
            return 0.0;
    }
}

__global__ void init_zero_state_batch_kernel(
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += stride) {
        state_batch[idx] = make_cuDoubleComplex((idx % state_len) == 0 ? 1.0 : 0.0, 0.0);
    }
}

__global__ void hadamard_batch_kernel(
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int qubit
) {
    const size_t pairs_per_sample = state_len >> 1;
    const size_t total_pairs = num_samples * pairs_per_sample;
    const size_t grid_stride = gridDim.x * blockDim.x;
    const size_t stride = 1ULL << qubit;
    const size_t block_size = stride << 1;
    const double norm = M_SQRT1_2;

    for (size_t global_pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_pair_idx < total_pairs;
         global_pair_idx += grid_stride) {
        const size_t sample_idx = global_pair_idx / pairs_per_sample;
        const size_t pair_idx = global_pair_idx % pairs_per_sample;

        const size_t block_idx = pair_idx / stride;
        const size_t pair_offset = pair_idx % stride;
        const size_t local_i = block_idx * block_size + pair_offset;
        const size_t local_j = local_i + stride;

        const size_t base = sample_idx << num_qubits;
        const size_t i = base + local_i;
        const size_t j = base + local_j;

        const cuDoubleComplex a = state_batch[i];
        const cuDoubleComplex b = state_batch[j];

        state_batch[i] = make_cuDoubleComplex(
            norm * (cuCreal(a) + cuCreal(b)),
            norm * (cuCimag(a) + cuCimag(b))
        );
        state_batch[j] = make_cuDoubleComplex(
            norm * (cuCreal(a) - cuCreal(b)),
            norm * (cuCimag(a) - cuCimag(b))
        );
    }
}

__global__ void diagonal_phase_batch_kernel(
    const double* __restrict__ features_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int entanglement_mode,
    double alpha,
    unsigned int map_func
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t basis_idx = global_idx & state_mask;
        const double* features = features_batch + sample_idx * num_qubits;

        double phase = 0.0;
        for (unsigned int i = 0; i < num_qubits; ++i) {
            if ((basis_idx >> i) & 1U) {
                phase += alpha * features[i];
            }
        }

        for (unsigned int i = 0; i < num_qubits; ++i) {
            for (unsigned int j = i + 1; j < num_qubits; ++j) {
                if (!pair_in_entanglement(i, j, num_qubits, entanglement_mode)) {
                    continue;
                }
                const unsigned int bit_i = (basis_idx >> i) & 1U;
                const unsigned int bit_j = (basis_idx >> j) & 1U;
                if ((bit_i ^ bit_j) == 0U) {
                    continue;
                }
                phase += alpha * pair_map_value(features[i], features[j], map_func);
            }
        }

        double sin_phase;
        double cos_phase;
        sincos(phase, &sin_phase, &cos_phase);

        const cuDoubleComplex value = state_batch[global_idx];
        state_batch[global_idx] = make_cuDoubleComplex(
            cuCreal(value) * cos_phase - cuCimag(value) * sin_phase,
            cuCreal(value) * sin_phase + cuCimag(value) * cos_phase
        );
    }
}

int run_zzfeaturemap_batch(
    const double* features_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int reps,
    unsigned int entanglement_mode,
    double alpha,
    unsigned int map_func,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0 || num_qubits < 2 || reps == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex* state_complex_d = static_cast<cuDoubleComplex*>(state_batch_d);
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const size_t total_elements = num_samples * state_len;
    const size_t blocks_needed = (total_elements + blockSize - 1) / blockSize;
    const size_t gridSize = (blocks_needed < MAX_GRID_BLOCKS) ? blocks_needed : MAX_GRID_BLOCKS;

    init_zero_state_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
        state_complex_d,
        num_samples,
        state_len
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }

    const size_t total_pairs = num_samples * (state_len >> 1);
    const size_t pair_blocks_needed = (total_pairs + blockSize - 1) / blockSize;
    const size_t pair_grid_size = (pair_blocks_needed < MAX_GRID_BLOCKS) ? pair_blocks_needed : MAX_GRID_BLOCKS;

    for (unsigned int rep = 0; rep < reps; ++rep) {
        for (unsigned int qubit = 0; qubit < num_qubits; ++qubit) {
            hadamard_batch_kernel<<<pair_grid_size, blockSize, 0, stream>>>(
                state_complex_d,
                num_samples,
                state_len,
                num_qubits,
                qubit
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return (int)err;
            }
        }

        diagonal_phase_batch_kernel<<<gridSize, blockSize, 0, stream>>>(
            features_batch_d,
            state_complex_d,
            num_samples,
            state_len,
            num_qubits,
            entanglement_mode,
            alpha,
            map_func
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return (int)err;
        }
    }

    return (int)cudaGetLastError();
}

}  // namespace

extern "C" {

int launch_zzfeaturemap_encode(
    const double* features_d,
    void* state_d,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int reps,
    unsigned int entanglement_mode,
    double alpha,
    unsigned int map_func,
    cudaStream_t stream
) {
    return run_zzfeaturemap_batch(
        features_d,
        state_d,
        1,
        state_len,
        num_qubits,
        reps,
        entanglement_mode,
        alpha,
        map_func,
        stream
    );
}

int launch_zzfeaturemap_encode_batch(
    const double* features_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int feature_len,
    unsigned int reps,
    unsigned int entanglement_mode,
    double alpha,
    unsigned int map_func,
    cudaStream_t stream
) {
    if (feature_len != num_qubits) {
        return cudaErrorInvalidValue;
    }

    return run_zzfeaturemap_batch(
        features_batch_d,
        state_batch_d,
        num_samples,
        state_len,
        num_qubits,
        reps,
        entanglement_mode,
        alpha,
        map_func,
        stream
    );
}

}  // extern "C"
