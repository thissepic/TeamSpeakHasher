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
#ifndef DEVICECONTEXT_H_
#define DEVICECONTEXT_H_

#include <cstdint>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

#include "IHasherDevice.h"

// forward declaration because of cyclic dependency
// between DeviceContext and TSHasherContext
class TSHasherContext;

class DeviceContext {
public:
  DeviceContext(std::string device_name,
    IHasherDevice* device,
    TSHasherContext* tshasherctx,
    size_t global_work_size,
    size_t local_work_size,
    uint8_t* h_results,
    std::string identitystring);

  std::string      device_name;
  IHasherDevice*   device;

  TSHasherContext* tshasherctx;

  size_t            global_work_size;
  size_t            local_work_size;

  uint8_t* h_results;

  volatile uint8_t      bestdifficulty;
  volatile uint64_t      bestdifficulty_counter;

  uint64_t      lastschedulediterations_total;
  uint64_t      lastscheduled_startcounter;
  uint64_t      schedulediterations_total;
  uint64_t      completediterations_total;
  uint64_t      completed_kernels;

  std::string      identitystring;

  void measureTime();

  std::chrono::duration<uint64_t, std::nano> getCurrentKernelRunningTime();

  double getAvgSpeed() const;

  std::chrono::duration<uint64_t, std::nano> getRecentMinTime() const;

  std::chrono::duration<uint64_t, std::nano> getRecentMaxTime() const;

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> laststarttime;
  std::vector<std::chrono::duration<uint64_t, std::nano>> recenttimes;
  std::vector<uint64_t> recentiterations;
  uint64_t timecounter;

  static const uint64_t NUM_TIME_MEASURMENTS;
  bool timer_started;
};

#endif
