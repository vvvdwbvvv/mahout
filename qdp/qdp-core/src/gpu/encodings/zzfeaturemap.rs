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

#![allow(unused_unsafe)]

use super::{QuantumEncoder, validate_qubit_count};
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::{GpuStateVector, Precision};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::map_allocation_error;
#[cfg(target_os = "linux")]
use cudarc::driver::{DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZzEntanglement {
    Full,
}

impl ZzEntanglement {
    #[must_use]
    fn as_u32(self) -> u32 {
        match self {
            Self::Full => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZzMapFunc {
    Qiskit,
}

impl ZzMapFunc {
    #[must_use]
    fn as_u32(self) -> u32 {
        match self {
            Self::Qiskit => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ZzFeatureMapConfig {
    reps: usize,
    entanglement: ZzEntanglement,
    alpha: f64,
    map_func: ZzMapFunc,
}

impl Default for ZzFeatureMapConfig {
    fn default() -> Self {
        Self {
            reps: 2,
            entanglement: ZzEntanglement::Full,
            alpha: 2.0,
            map_func: ZzMapFunc::Qiskit,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ZzFeatureMapEncoder {
    config: ZzFeatureMapConfig,
}

impl ZzFeatureMapEncoder {
    fn validate_shape(&self, data_len: usize, num_qubits: usize) -> Result<()> {
        validate_qubit_count(num_qubits)?;
        if num_qubits < 2 {
            return Err(MahoutError::InvalidInput(
                "zzfeaturemap requires at least 2 qubits to match the Qiskit reference circuit"
                    .to_string(),
            ));
        }
        if data_len != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "zzfeaturemap expects {} features (one per qubit), got {}",
                num_qubits, data_len
            )));
        }
        Ok(())
    }
}

impl QuantumEncoder for ZzFeatureMapEncoder {
    fn encode(
        &self,
        #[cfg(target_os = "linux")] device: &Arc<CudaDevice>,
        #[cfg(not(target_os = "linux"))] _device: &Arc<CudaDevice>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        self.validate_input(data, num_qubits)?;
        let state_len = 1 << num_qubits;

        #[cfg(target_os = "linux")]
        {
            let input_bytes = std::mem::size_of_val(data);
            let features_gpu = {
                crate::profile_scope!("GPU::H2D_ZzFeatureMapData");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(
                        input_bytes,
                        "zzfeaturemap input upload",
                        Some(num_qubits),
                        e,
                    )
                })?
            };

            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits, Precision::Float64)?
            };
            let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "State vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_zzfeaturemap_encode(
                        *features_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        state_len,
                        num_qubits as u32,
                        self.config.reps as u32,
                        self.config.entanglement.as_u32(),
                        self.config.alpha,
                        self.config.map_func.as_u32(),
                        std::ptr::null_mut(),
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "zzfeaturemap encoding kernel failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }

            device.synchronize().map_err(|e| {
                MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e))
            })?;

            Ok(state_vector)
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda(
                "CUDA unavailable (non-Linux stub)".to_string(),
            ))
        }
    }

    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("ZzFeatureMapEncoder::encode_batch");

        self.validate_shape(sample_size, num_qubits)?;
        if batch_data.len() != num_samples * sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "Batch data length {} doesn't match num_samples {} * sample_size {}",
                batch_data.len(),
                num_samples,
                sample_size
            )));
        }

        for (i, &val) in batch_data.iter().enumerate() {
            if !val.is_finite() {
                let sample_idx = i / sample_size;
                let feature_idx = i % sample_size;
                return Err(MahoutError::InvalidInput(format!(
                    "Sample {} feature {} must be finite, got {}",
                    sample_idx, feature_idx, val
                )));
            }
        }

        let state_len = 1 << num_qubits;
        let batch_state_vector =
            GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?;

        let input_bytes = std::mem::size_of_val(batch_data);
        let features_gpu = {
            crate::profile_scope!("GPU::H2D_BatchZzFeatureMapData");
            device.htod_sync_copy(batch_data).map_err(|e| {
                map_allocation_error(
                    input_bytes,
                    "zzfeaturemap batch upload",
                    Some(num_qubits),
                    e,
                )
            })?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        let ret = unsafe {
            qdp_kernels::launch_zzfeaturemap_encode_batch(
                *features_gpu.device_ptr() as *const f64,
                state_ptr as *mut c_void,
                num_samples,
                state_len,
                num_qubits as u32,
                sample_size as u32,
                self.config.reps as u32,
                self.config.entanglement.as_u32(),
                self.config.alpha,
                self.config.map_func.as_u32(),
                std::ptr::null_mut(),
            )
        };
        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "Batch zzfeaturemap encoding kernel failed: {} ({})",
                ret,
                cuda_error_to_string(ret)
            )));
        }

        device
            .synchronize()
            .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;

        Ok(batch_state_vector)
    }

    #[cfg(target_os = "linux")]
    unsafe fn encode_from_gpu_ptr(
        &self,
        device: &Arc<CudaDevice>,
        input_d: *const c_void,
        input_len: usize,
        num_qubits: usize,
        stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        self.validate_shape(input_len, num_qubits)?;

        let feature_validation_buffer = {
            crate::profile_scope!("GPU::ZzFeatureMapFiniteCheck");
            let mut buffer = device.alloc_zeros::<f64>(1).map_err(|e| {
                MahoutError::MemoryAllocation(format!(
                    "Failed to allocate zzfeaturemap validation buffer: {:?}",
                    e
                ))
            })?;
            let ret = unsafe {
                qdp_kernels::launch_l2_norm(
                    input_d as *const f64,
                    input_len,
                    *buffer.device_ptr_mut() as *mut f64,
                    stream,
                )
            };
            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "zzfeaturemap validation norm kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
            buffer
        };
        let host_norm = device
            .dtoh_sync_copy(&feature_validation_buffer)
            .map_err(|e| {
                MahoutError::Cuda(format!(
                    "Failed to copy zzfeaturemap validation result to host: {:?}",
                    e
                ))
            })?;
        if host_norm.iter().any(|v| !v.is_finite()) {
            return Err(MahoutError::InvalidInput(
                "zzfeaturemap input contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let state_len = 1 << num_qubits;
        let state_vector = GpuStateVector::new(device, num_qubits, Precision::Float64)?;
        let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "State vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        let ret = unsafe {
            qdp_kernels::launch_zzfeaturemap_encode(
                input_d as *const f64,
                state_ptr as *mut c_void,
                state_len,
                num_qubits as u32,
                self.config.reps as u32,
                self.config.entanglement.as_u32(),
                self.config.alpha,
                self.config.map_func.as_u32(),
                stream,
            )
        };
        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "zzfeaturemap encoding kernel failed with CUDA error code: {} ({})",
                ret,
                cuda_error_to_string(ret)
            )));
        }

        crate::gpu::cuda_sync::sync_cuda_stream(stream, "CUDA stream synchronize failed")?;
        Ok(state_vector)
    }

    #[cfg(target_os = "linux")]
    unsafe fn encode_batch_from_gpu_ptr(
        &self,
        device: &Arc<CudaDevice>,
        input_batch_d: *const c_void,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        self.validate_shape(sample_size, num_qubits)?;

        let validation_buffer = {
            crate::profile_scope!("GPU::ZzFeatureMapFiniteCheckBatch");
            let mut buffer = device.alloc_zeros::<f64>(num_samples).map_err(|e| {
                MahoutError::MemoryAllocation(format!(
                    "Failed to allocate zzfeaturemap validation buffer: {:?}",
                    e
                ))
            })?;
            let ret = unsafe {
                qdp_kernels::launch_l2_norm_batch(
                    input_batch_d as *const f64,
                    num_samples,
                    sample_size,
                    *buffer.device_ptr_mut() as *mut f64,
                    stream,
                )
            };
            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "zzfeaturemap validation norm kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
            buffer
        };
        let host_norms = device.dtoh_sync_copy(&validation_buffer).map_err(|e| {
            MahoutError::Cuda(format!(
                "Failed to copy zzfeaturemap batch validation norms to host: {:?}",
                e
            ))
        })?;
        if host_norms.iter().any(|v| !v.is_finite()) {
            return Err(MahoutError::InvalidInput(
                "zzfeaturemap batch contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let state_len = 1 << num_qubits;
        let batch_state_vector =
            GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?;
        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        let ret = unsafe {
            qdp_kernels::launch_zzfeaturemap_encode_batch(
                input_batch_d as *const f64,
                state_ptr as *mut c_void,
                num_samples,
                state_len,
                num_qubits as u32,
                sample_size as u32,
                self.config.reps as u32,
                self.config.entanglement.as_u32(),
                self.config.alpha,
                self.config.map_func.as_u32(),
                stream,
            )
        };
        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "Batch zzfeaturemap encoding kernel failed: {} ({})",
                ret,
                cuda_error_to_string(ret)
            )));
        }

        crate::gpu::cuda_sync::sync_cuda_stream(stream, "CUDA stream synchronize failed")?;
        Ok(batch_state_vector)
    }

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        self.validate_shape(data.len(), num_qubits)?;
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "zzfeaturemap feature at index {} must be finite, got {}",
                    i, val
                )));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "zzfeaturemap"
    }

    fn description(&self) -> &'static str {
        "Qiskit-style ZZFeatureMap encoding with configurable reps, entanglement, alpha, and map_func"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_single_qubit_zzfeaturemap() {
        let encoder = ZzFeatureMapEncoder::default();
        let err = encoder
            .validate_input(&[0.5], 1)
            .expect_err("single-qubit zzfeaturemap should be rejected");
        assert!(matches!(err, MahoutError::InvalidInput(_)));
    }

    #[test]
    fn default_config_matches_qiskit_defaults() {
        let config = ZzFeatureMapConfig::default();
        assert_eq!(config.reps, 2);
        assert_eq!(config.entanglement, ZzEntanglement::Full);
        assert_eq!(config.alpha, 2.0);
        assert_eq!(config.map_func, ZzMapFunc::Qiskit);
    }
}
