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
#ifndef IHASHERDEVICE_H_
#define IHASHERDEVICE_H_

#include <cstdint>
#include <string>

enum class Backend {
  OPENCL,
  CUDA
};

class IHasherDevice {
public:
  virtual ~IHasherDevice() {}

  // Device identity
  virtual std::string getDeviceName() const = 0;
  virtual std::string getDeviceIdentifier(uint32_t device_id) const = 0;
  virtual uint32_t getMaxComputeUnits() const = 0;
  virtual bool isGPU() const = 0;

  // Kernel compilation (no-op for CUDA since nvcc compiles at build time)
  virtual bool compileKernel() = 0;

  // Work group size queries (needed for tuning)
  virtual size_t getMaxLocalWorkSize() const = 0;
  virtual size_t getMaxWorkItemSize() const = 0;

  // Buffer management
  virtual void allocateBuffers(size_t global_work_size,
                               const std::string& identity) = 0;

  // Kernel execution: sets arguments and launches
  virtual void setArgsAndLaunch(uint64_t startcounter,
                                uint32_t iterations,
                                uint8_t targetdifficulty,
                                uint32_t identity_length,
                                size_t global_work_size,
                                size_t local_work_size,
                                bool use_kernel2) = 0;

  // Wait for kernel to complete
  virtual void finish() = 0;

  // Copy results from device to host
  virtual void readResults(uint8_t* h_results, size_t global_work_size) = 0;

  // Create an independent context for tuning (caller owns the pointer)
  virtual IHasherDevice* createTuningContext() const = 0;
};

#endif
