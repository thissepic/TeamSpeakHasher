/*
Copyright (c) 2017 landave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "CUDADevice.h"
#include "CUDAKernel.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <regex>
#include <string>

CUDADevice::CUDADevice(int cuda_device_id)
    : cuda_device_id_(cuda_device_id),
      stream_(nullptr),
      d_results_(nullptr),
      d_identity_(nullptr),
      identity_length_(0),
      alloc_global_work_size_(0) {
  cudaSetDevice(cuda_device_id_);
  cudaGetDeviceProperties(&props_, cuda_device_id_);
  cudaStreamCreate(&stream_);
}

CUDADevice::~CUDADevice() {
  if (d_results_) {
    cudaFree(d_results_);
  }
  if (d_identity_) {
    cudaFree(d_identity_);
  }
  cudaStreamDestroy(stream_);
}

std::string CUDADevice::getDeviceName() const {
  std::string name(props_.name);
  // Trim leading and trailing whitespace
  name = std::regex_replace(name, std::regex("^\\s+"), std::string(""));
  name = std::regex_replace(name, std::regex("\\s+$"), std::string(""));
  return name;
}

std::string CUDADevice::getDeviceIdentifier(uint32_t device_id) const {
  std::string name = getDeviceName();
  std::string id = "CUDA_" + name + "_SM"
      + std::to_string(props_.major) + std::to_string(props_.minor)
      + "_" + std::to_string(device_id);
  // Replace non-word characters with underscore
  id = std::regex_replace(id, std::regex("[^a-zA-Z0-9_]"), std::string("_"));
  return id;
}

uint32_t CUDADevice::getMaxComputeUnits() const {
  return static_cast<uint32_t>(props_.multiProcessorCount);
}

bool CUDADevice::isGPU() const {
  return true;
}

bool CUDADevice::compileKernel() {
  // CUDA kernels are compiled at build time by nvcc.
  // Just ensure we are bound to the correct device.
  cudaSetDevice(cuda_device_id_);
  return true;
}
/*
size_t CUDADevice::getMaxLocalWorkSize() const
{
    return 256; // Hardcode zum Testen, um den Tuner nicht zu blockieren!
}
*/
size_t CUDADevice::getMaxLocalWorkSize() const {
  // Use occupancy API to find the actual max block size that accounts for
  // register pressure of our SHA1 kernels (not just the device maximum).
  int minGrid1 = 0, block1 = 0;
  int minGrid2 = 0, block2 = 0;
  cudaOccupancyMaxPotentialBlockSize(&minGrid1, &block1,
                                     TeamSpeakHasher_cuda, 0, 0);
  cudaOccupancyMaxPotentialBlockSize(&minGrid2, &block2,
                                     TeamSpeakHasher2_cuda, 0, 0);
  int maxBlock = std::min(block1, block2);
  if (maxBlock <= 0) {
    // Fallback: conservative default if occupancy query fails
    return 256;
  }
  return static_cast<size_t>(maxBlock);
}

size_t CUDADevice::getMaxWorkItemSize() const {
  return static_cast<size_t>(props_.maxThreadsDim[0]);
}

void CUDADevice::allocateBuffers(size_t global_work_size,
                                 const std::string& identity) {
  cudaSetDevice(cuda_device_id_);

  alloc_global_work_size_ = global_work_size;
  identity_length_ = identity.size();

  cudaMalloc(&d_results_, global_work_size * sizeof(unsigned char));
  cudaMemsetAsync(d_results_, 0, global_work_size * sizeof(unsigned char),
                  stream_);

  cudaMalloc(&d_identity_, identity.size());
  cudaMemcpyAsync(d_identity_, identity.c_str(), identity.size(),
                  cudaMemcpyHostToDevice, stream_);

  cudaStreamSynchronize(stream_);
}

void CUDADevice::setArgsAndLaunch(uint64_t startcounter,
                                  uint32_t iterations,
                                  uint8_t targetdifficulty,
                                  uint32_t identity_length,
                                  size_t global_work_size,
                                  size_t local_work_size,
                                  bool use_kernel2) {
  cudaSetDevice(cuda_device_id_);

  dim3 block(static_cast<unsigned int>(local_work_size));
  dim3 grid(static_cast<unsigned int>(global_work_size / local_work_size));

  if (use_kernel2) {
    TeamSpeakHasher2_cuda<<<grid, block, 0, stream_>>>(
        startcounter, iterations, targetdifficulty,
        d_identity_, identity_length, d_results_);
  } else {
    TeamSpeakHasher_cuda<<<grid, block, 0, stream_>>>(
        startcounter, iterations, targetdifficulty,
        d_identity_, identity_length, d_results_);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
              << std::endl;
  }
}

void CUDADevice::finish() {
  cudaSetDevice(cuda_device_id_);
  cudaStreamSynchronize(stream_);
}

void CUDADevice::readResults(uint8_t* h_results, size_t global_work_size) {
  cudaSetDevice(cuda_device_id_);
  cudaMemcpy(h_results, d_results_, global_work_size * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
}

IHasherDevice* CUDADevice::createTuningContext() const {
  return new CUDADevice(cuda_device_id_);
}
