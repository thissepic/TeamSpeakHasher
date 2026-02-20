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
#ifndef CUDADEVICE_H_
#define CUDADEVICE_H_

#include "IHasherDevice.h"
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

class CUDADevice : public IHasherDevice {
public:
  CUDADevice(int cuda_device_id);
  ~CUDADevice() override;

  std::string getDeviceName() const override;
  std::string getDeviceIdentifier(uint32_t device_id) const override;
  uint32_t getMaxComputeUnits() const override;
  bool isGPU() const override;

  bool compileKernel() override;
  size_t getMaxLocalWorkSize() const override;
  size_t getMaxWorkItemSize() const override;

  void allocateBuffers(size_t global_work_size, const std::string& identity) override;

  void setArgsAndLaunch(uint64_t startcounter, uint32_t iterations,
                        uint8_t targetdifficulty, uint32_t identity_length,
                        size_t global_work_size, size_t local_work_size,
                        bool use_kernel2) override;

  void finish() override;
  void readResults(uint8_t* h_results, size_t global_work_size) override;
  IHasherDevice* createTuningContext() const override;

private:
  int cuda_device_id_;
  cudaDeviceProp props_;
  cudaStream_t stream_;
  unsigned char* d_results_;
  unsigned char* d_identity_;
  size_t identity_length_;
  size_t alloc_global_work_size_;
};

#endif
