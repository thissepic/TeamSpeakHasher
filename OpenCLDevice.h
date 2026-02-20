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
#ifndef OPENCLDEVICE_H_
#define OPENCLDEVICE_H_

#include "IHasherDevice.h"

#include <cstdint>
#include <string>

#include <CL/cl.hpp>

class OpenCLDevice : public IHasherDevice {
public:
  OpenCLDevice(cl::Device device, uint32_t vendor_id);
  ~OpenCLDevice() override;

  std::string getDeviceName() const override;
  std::string getDeviceIdentifier(uint32_t device_id) const override;
  uint32_t getMaxComputeUnits() const override;
  bool isGPU() const override;

  bool compileKernel() override;
  size_t getMaxLocalWorkSize() const override;
  size_t getMaxWorkItemSize() const override;

  void allocateBuffers(size_t global_work_size,
                       const std::string& identity) override;

  void setArgsAndLaunch(uint64_t startcounter,
                        uint32_t iterations,
                        uint8_t targetdifficulty,
                        uint32_t identity_length,
                        size_t global_work_size,
                        size_t local_work_size,
                        bool use_kernel2) override;

  void finish() override;
  void readResults(uint8_t* h_results, size_t global_work_size) override;
  IHasherDevice* createTuningContext() const override;

private:
  cl::Device device_;
  cl::Context context_;
  cl::Program program_;
  cl::Kernel kernel_;
  cl::Kernel kernel2_;
  cl::CommandQueue queue_;
  cl::Buffer d_results_;
  cl::Buffer d_identity_;

  uint32_t vendor_id_;
  std::string device_name_;
  cl_uint max_compute_units_;
  cl_device_type devicetype_;

  static const char* KERNEL_CODE;
  static const char* KERNEL_NAME;
  static const char* KERNEL_NAME2;
  static const char* BUILD_OPTS_BASE;
};

#endif
