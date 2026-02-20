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
#include "DeviceContext.h"

#include <cstdint>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

#include "TSHasherContext.h"


const uint64_t DeviceContext::NUM_TIME_MEASURMENTS = 32;


DeviceContext::DeviceContext(std::string device_name,
  IHasherDevice* device,
  TSHasherContext* tshasherctx,
  size_t global_work_size,
  size_t local_work_size,
  uint8_t* h_results,
  std::string identitystring) :
  device_name(std::move(device_name)),
  device(device),
  tshasherctx(tshasherctx),
  global_work_size(global_work_size),
  local_work_size(local_work_size),
  h_results(h_results),
  identitystring(std::move(identitystring)),
  recenttimes(NUM_TIME_MEASURMENTS),
  recentiterations(NUM_TIME_MEASURMENTS) {
  bestdifficulty = 0;
  bestdifficulty_counter = 0;
  lastschedulediterations_total = 0;
  lastscheduled_startcounter = 0;
  schedulediterations_total = 0;
  completediterations_total = 0;
  completed_kernels = 0;
  timer_started = false;
  timecounter = 0;
}

void DeviceContext::measureTime() {
  auto currenttime = std::chrono::high_resolution_clock::now();
  if (timer_started) {
    auto timeidx = timecounter % NUM_TIME_MEASURMENTS;
    recenttimes[timeidx] = currenttime - laststarttime;
    recentiterations[timeidx] = lastschedulediterations_total;
    timecounter++;
  }
  laststarttime = currenttime;
  timer_started = true;
}

std::chrono::duration<uint64_t, std::nano> DeviceContext::getCurrentKernelRunningTime() {
  return std::chrono::high_resolution_clock::now() - laststarttime;
}

double DeviceContext::getAvgSpeed() const {
  using namespace std::chrono;
  size_t len = std::min(timecounter, NUM_TIME_MEASURMENTS);
  auto totaliterations = std::accumulate(recentiterations.begin(), recentiterations.begin() + len, (uint64_t)0);
  auto totaltime = std::accumulate(recenttimes.begin(), recenttimes.begin() + len, steady_clock::duration::zero());
  auto seconds = duration_cast<nanoseconds>(totaltime).count() / 1e9;
  return totaliterations / seconds;
}

std::chrono::duration<uint64_t, std::nano> DeviceContext::getRecentMinTime() const {
  return *std::min_element(recenttimes.begin(), recenttimes.end());
}

std::chrono::duration<uint64_t, std::nano> DeviceContext::getRecentMaxTime() const {
  return *std::max_element(recenttimes.begin(), recenttimes.end());
}
