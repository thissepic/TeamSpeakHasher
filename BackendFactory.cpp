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
#include "BackendFactory.h"

#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_OPENCL
#include "OpenCLDevice.h"
#include <CL/cl.hpp>

#define VENDOR_ID_GENERIC 0
#define VENDOR_ID_AMD 1
#define VENDOR_ID_NV (1<<5)
#endif

#ifdef HAVE_CUDA
#include "CUDADevice.h"
#include <cuda_runtime.h>
#endif


Backend parseBackend(const std::string& str) {
  if (str == "cuda" || str == "CUDA") return Backend::CUDA;
  if (str == "opencl" || str == "OpenCL" || str == "OPENCL") return Backend::OPENCL;
  return Backend::OPENCL;
}

std::vector<IHasherDevice*> createDevices(Backend backend) {
  std::vector<IHasherDevice*> devices;

  if (backend == Backend::OPENCL) {
#ifdef HAVE_OPENCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (uint32_t i = 0; i < platforms.size(); i++) {
      auto& platform = platforms[i];
      auto platform_vendor = std::string(platform.getInfo<CL_PLATFORM_VENDOR>().c_str());

      uint32_t platform_vendor_id;
      if (platform_vendor == "Advanced Micro Devices, Inc." || platform_vendor == "AuthenticAMD") {
        platform_vendor_id = VENDOR_ID_AMD;
      }
      else if (platform_vendor == "NVIDIA Corporation") {
        platform_vendor_id = VENDOR_ID_NV;
      }
      else {
        platform_vendor_id = VENDOR_ID_GENERIC;
      }

      std::vector<cl::Device> cl_devices;
      platform.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices);

      for (auto& cl_device : cl_devices) {
        devices.push_back(new OpenCLDevice(cl_device, platform_vendor_id));
      }
    }
#else
    std::cerr << "Error: OpenCL support was not compiled in." << std::endl;
    exit(-1);
#endif
  }
  else if (backend == Backend::CUDA) {
#ifdef HAVE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
      devices.push_back(new CUDADevice(i));
    }
#else
    std::cerr << "Error: CUDA support was not compiled in." << std::endl;
    exit(-1);
#endif
  }

  return devices;
}
