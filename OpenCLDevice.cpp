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
#include "OpenCLDevice.h"

#include <cstdint>
#include <cstring>

#include <iostream>
#include <regex>
#include <string>

#include <CL/cl.hpp>

#include "OpenCLKernel.h"

#define VENDOR_ID_GENERIC 0
#define VENDOR_ID_AMD 1
#define VENDOR_ID_NV (1<<5)

const char* OpenCLDevice::KERNEL_NAME = "TeamSpeakHasher";
const char* OpenCLDevice::KERNEL_NAME2 = "TeamSpeakHasher2";
const char* OpenCLDevice::BUILD_OPTS_BASE = "-I . -I src";


OpenCLDevice::OpenCLDevice(cl::Device device, uint32_t vendor_id)
  : device_(device),
    vendor_id_(vendor_id),
    max_compute_units_(0),
    devicetype_(0) {

  devicetype_ = device_.getInfo<CL_DEVICE_TYPE>();
  max_compute_units_ = device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

  std::string raw_name = std::string(device_.getInfo<CL_DEVICE_NAME>().c_str());
  auto regex = std::regex("^ +| +$|( ) +");
  device_name_ = std::regex_replace(raw_name, regex, "$1");

  context_ = cl::Context({ device_ });
}

OpenCLDevice::~OpenCLDevice() {
}

std::string OpenCLDevice::getDeviceName() const {
  return device_name_;
}

std::string OpenCLDevice::getDeviceIdentifier(uint32_t device_id) const {
  auto devicename = std::string(device_.getInfo<CL_DEVICE_NAME>().c_str())
    + "_" + std::string(device_.getInfo<CL_DEVICE_VENDOR>().c_str())
    + "_" + std::to_string(device_.getInfo<CL_DEVICE_VENDOR_ID>())
    + "_" + std::string(device_.getInfo<CL_DEVICE_VERSION>().c_str())
    + "_" + std::string(device_.getInfo<CL_DRIVER_VERSION>().c_str())
    + "_" + std::to_string(device_id);

  const auto target = std::regex{ R"([^\w])" };
  const auto replacement = std::string{ "_" };

  return std::regex_replace(devicename, target, replacement);
}

uint32_t OpenCLDevice::getMaxComputeUnits() const {
  return max_compute_units_;
}

bool OpenCLDevice::isGPU() const {
  return (devicetype_ & CL_DEVICE_TYPE_GPU) != 0;
}

bool OpenCLDevice::compileKernel() {
  cl::Program::Sources sources;
  sources.push_back({ KERNEL_CODE, strlen(KERNEL_CODE) });
  program_ = cl::Program(context_, sources);

  cl_uint vector_width = device_.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>();

  cl_uint sm_minor = 0;
  cl_uint sm_major = 0;

  if (vendor_id_ == VENDOR_ID_NV) {
    sm_minor = device_.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>();
    sm_major = device_.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>();
  }

  char build_opts[2048] = { 0 };
  snprintf(build_opts,
    sizeof(build_opts) - 1,
    "%s -D VENDOR_ID=%u "
    "-D CUDA_ARCH=%u "
    "-D VECT_SIZE=%u "
    "-D DEVICE_TYPE=%u "
    "-D _unroll "
    "-cl-std=CL1.2 -w",
    BUILD_OPTS_BASE,
    vendor_id_,
    (sm_major * 100) + sm_minor,
    vector_width,
    (uint32_t)devicetype_);

  if (program_.build({ device_ }, build_opts) != CL_SUCCESS) {
    std::cout << " Error building: " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
    return false;
  }

  kernel_ = cl::Kernel(program_, KERNEL_NAME);
  kernel2_ = cl::Kernel(program_, KERNEL_NAME2);
  queue_ = cl::CommandQueue(context_, device_);

  return true;
}

size_t OpenCLDevice::getMaxLocalWorkSize() const {
  size_t max_local = 0;
  kernel_.getWorkGroupInfo<size_t>(device_, CL_KERNEL_WORK_GROUP_SIZE, &max_local);
  return max_local;
}

size_t OpenCLDevice::getMaxWorkItemSize() const {
  return device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
}

void OpenCLDevice::allocateBuffers(size_t global_work_size,
                                   const std::string& identity) {
  const size_t size_results = global_work_size * sizeof(uint8_t);

  d_results_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, size_results);
  d_identity_ = cl::Buffer(context_, CL_MEM_READ_ONLY, identity.size());

  auto h_results = new uint8_t[global_work_size]();
  queue_.enqueueWriteBuffer(d_results_, CL_TRUE, 0, size_results, h_results);
  delete[] h_results;

  const cl_uint identity_length = (cl_uint)identity.size();
  queue_.enqueueWriteBuffer(d_identity_, CL_TRUE, 0, identity_length, identity.c_str());
}

void OpenCLDevice::setArgsAndLaunch(uint64_t startcounter,
                                    uint32_t iterations,
                                    uint8_t targetdifficulty,
                                    uint32_t identity_length,
                                    size_t global_work_size,
                                    size_t local_work_size,
                                    bool use_kernel2) {
  cl::Kernel& kernel = use_kernel2 ? kernel2_ : kernel_;

  cl_int err;
  err = kernel.setArg(0, (cl_ulong)startcounter);
  err |= kernel.setArg(1, (cl_uint)iterations);
  err |= kernel.setArg(2, (cl_uchar)targetdifficulty);
  err |= kernel.setArg(3, d_identity_);
  err |= kernel.setArg(4, (cl_uint)identity_length);
  err |= kernel.setArg(5, d_results_);

  if (err != CL_SUCCESS) {
    std::cout << "A critical error occurred while setting kernel arguments." << std::endl;
    exit(-1);
  }

  err = queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
    cl::NDRange(global_work_size),
    cl::NDRange(local_work_size),
    NULL);

  if (err != CL_SUCCESS) {
    std::cout << "A critical error occurred while enqueuing the kernel." << std::endl;
    exit(-1);
  }
}

void OpenCLDevice::finish() {
  queue_.finish();
}

void OpenCLDevice::readResults(uint8_t* h_results, size_t global_work_size) {
  const size_t size_results = global_work_size * sizeof(uint8_t);
  queue_.enqueueReadBuffer(d_results_, CL_TRUE, 0, size_results, h_results);
}

IHasherDevice* OpenCLDevice::createTuningContext() const {
  return new OpenCLDevice(device_, vendor_id_);
}
