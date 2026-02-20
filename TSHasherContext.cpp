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
#include "TSHasherContext.h"

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "BackendFactory.h"
#include "Config.h"
#include "DeviceContext.h"
#include "sha1.h"
#include "Table.h"
#include "TSUtil.h"
#include "TimerKiller.h"
#include "TunedParameters.h"


const size_t TSHasherContext::MAX_DEVICES = 16;
const size_t TSHasherContext::DEV_DEFAULT_LOCAL_WORK_SIZE = 32;
const size_t TSHasherContext::DEV_DEFAULT_GLOBAL_WORK_SIZE = 64 * 4096;
const uint64_t TSHasherContext::MAX_GLOBALLOCAL_RATIO = (1 << 16);
const size_t TSHasherContext::KERNEL_STD_ITERATIONS = 1024;


#if defined(_WIN32) || defined(_WIN64)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <Windows.h>
void TSHasherContext::clearConsole() {
  system("cls");
}

#else
#include <signal.h>

void TSHasherContext::clearConsole() {
  system("clear");
}
#endif


TSHasherContext::TSHasherContext(std::string identity,
  uint64_t startcounter,
  uint64_t bestcounter,
  uint64_t throttlefactor,
  Backend backend) :
  startcounter(startcounter),
  global_bestdifficulty(TSUtil::getDifficulty(identity, bestcounter)),
  global_bestdifficulty_counter(bestcounter),
  identity(identity),
  throttlefactor(throttlefactor) {
  MIN_TARGET_DIFFICULTY = 34;

  auto raw_devices = createDevices(backend);

  if (raw_devices.empty()) {
    std::cerr << "Error: No devices have been found." << std::endl;
    exit(-1);
  }

  for (uint32_t device_id = 0; device_id < raw_devices.size(); device_id++) {
    IHasherDevice* dev = raw_devices[device_id];
    owned_devices.push_back(dev);

    if (!dev->compileKernel()) {
      std::cerr << "Error: Failed to compile kernel for device " << dev->getDeviceName() << std::endl;
      exit(1);
    }

    std::string device_name = dev->getDeviceName();
    uint32_t max_compute_units = dev->getMaxComputeUnits();

    std::cout << "Found new device #" << device_id << ": " << device_name << ", " << max_compute_units << " compute units" << std::endl;

    size_t global_work_size = DEV_DEFAULT_GLOBAL_WORK_SIZE;
    size_t local_work_size = DEV_DEFAULT_LOCAL_WORK_SIZE;
    auto bestgloballocal = tune(dev, device_id);
    global_work_size = std::max(bestgloballocal.first / throttlefactor, bestgloballocal.second);
    local_work_size = bestgloballocal.second;

    // allocate device buffers and upload identity
    dev->allocateBuffers(global_work_size, identity);

    // host memory
    auto h_results = new uint8_t[global_work_size]();

    DeviceContext dev_ctx(device_name, dev, this,
      global_work_size, local_work_size, h_results, identity);

    dev_ctxs.push_back(dev_ctx);
  }

  Config::store();
}

TSHasherContext::~TSHasherContext() {
  for (auto* dev : owned_devices) {
    delete dev;
  }
}


std::pair<uint64_t, uint64_t> TSHasherContext::tune(IHasherDevice* device,
  uint32_t device_id) {
  //return { 2097152, 32 };
  using namespace std::chrono;
  std::string deviceidentifier = device->getDeviceIdentifier(device_id);
  auto conf = Config::tuned.find(deviceidentifier);
  if (conf != Config::tuned.end()) {
    return { conf->second.globalworksize, conf->second.localworksize };
  }

  std::cout << "  Tuning... " << std::endl;

  uint64_t besttime = UINT64_MAX;
  std::pair<uint64_t, uint64_t> result;
  uint64_t tunestartcounter = 10000000000000000000ULL;
  const uint8_t tunetargetdifficulty = 60;
  const std::string tuneidentity = "AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPPQQQRRRSSSTTTUUUVVVWWWXXXYYYZZZAAABBBCCCDDDEEEFFFGGG";
  const size_t identity_length = tuneidentity.size();

  // create a separate context for tuning
  IHasherDevice* tune_dev = device->createTuningContext();
  if (!tune_dev->compileKernel()) {
    std::cerr << "Error: Failed to compile kernel for tuning." << std::endl;
    exit(1);
  }

  size_t max_local_worksize = tune_dev->getMaxLocalWorkSize();
  max_local_worksize = std::min(max_local_worksize, tune_dev->getMaxWorkItemSize());

  const size_t max_global_work_size = MAX_GLOBALLOCAL_RATIO * max_local_worksize;
  //std::cout << std::endl << std::endl << "  Max Local Work Size=" << max_local_worksize << 
  //    "  Max Global Work Size=" << max_global_work_size << std::endl;
  tune_dev->allocateBuffers(max_global_work_size, tuneidentity);

  uint8_t* tune_h_results = new uint8_t[max_global_work_size]();

  std::cout << "  ";
  const size_t start_local_worksize = max_local_worksize > 128 ? 16 : 1;
  const size_t end_local_worksize = max_local_worksize;
  const size_t repetitions = 30;
  const size_t warmup_runs = 5;
  const size_t max_time_per_kernel_ms = 150;

  for (size_t localsize = start_local_worksize;
    localsize <= end_local_worksize;
    localsize *= 2) {
    size_t globalsize = localsize;
    duration<uint64_t, std::nano> time = steady_clock::duration::zero();
    while (globalsize <= max_global_work_size && duration_cast<milliseconds>(time / repetitions).count() < max_time_per_kernel_ms) {
      time_point<high_resolution_clock> starttime;

      for (size_t q = 0; q < repetitions + warmup_runs; q++) {
        if (q == warmup_runs) {
          starttime = high_resolution_clock::now();
        }

        tune_dev->setArgsAndLaunch(tunestartcounter, (uint32_t)KERNEL_STD_ITERATIONS,
          tunetargetdifficulty, (uint32_t)identity_length,
          globalsize, localsize, false);

        tune_dev->finish();
        tune_dev->readResults(tune_h_results, globalsize);
      }

      time = high_resolution_clock::now() - starttime;
      auto time_ns = duration_cast<nanoseconds>(time).count();
      uint64_t time_norm = time_ns / globalsize;

      std::cout << ".";

      if (duration_cast<milliseconds>(time / repetitions).count() < max_time_per_kernel_ms && time_norm < besttime) {
        besttime = time_norm;
        result.first = globalsize;
        result.second = localsize;
      }
      globalsize *= 2;
    }
  }

  delete[] tune_h_results;
  delete tune_dev;

  std::string devicename = device->getDeviceName();
  TunedParameters tunedparams(devicename, deviceidentifier, result.second, result.first);
  Config::tuned.insert(std::make_pair(deviceidentifier, tunedparams));

  std::cout << std::endl << std::endl << "  Tuning found global_work_size=" << result.first
    << ", local_work_size=" << result.second << " to be optimal." << std::endl;
  return result;
}

void TSHasherContext::compute() {
  TSHasherContext::starttime = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> threads;
  for (uint32_t device_id = 0; device_id < dev_ctxs.size(); device_id++) {
    DeviceContext* dev_ctx = &dev_ctxs[device_id];
    std::thread t([dev_ctx]() -> void { run_kernel_loop(dev_ctx); });
    threads.push_back(std::move(t));
  }

  // this loops until stopped
  TSHasherContext::printinfo(dev_ctxs);

  for (std::thread& t : threads) {
    t.join();
  }
}


std::string TSHasherContext::getFormattedDouble(double x) {
  if (x > 1000000000000) { return std::to_string(x / 1000000000000) + " T"; }
  if (x > 1000000000) { return std::to_string(x / 1000000000) + " G"; }
  if (x > 1000000) { return std::to_string(x / 1000000) + " M"; }
  return std::to_string(x) + " ";
}

std::string TSHasherContext::getFormattedDuration(double seconds) {
  //round towards zero
  uint64_t second = static_cast<uint64_t>(seconds);
  std::string out = "";
  if (second / 86400 > 0) {
    uint64_t days = second / 86400;
    out += std::to_string(days) + " days ";
    second -= days * 86400;
  }
  if (second / 3600 > 0) {
    uint64_t h = second / 3600;
    out += std::to_string(h) + " h ";
    second -= h * 3600;
  }
  if (second / 60 > 0) {
    uint64_t min = second / 60;
    out += std::to_string(min) + " min ";
    second -= min * 60;
  }
  out += std::to_string(second) + " s";
  return out;
}

void TSHasherContext::printinfo(const std::vector<DeviceContext>& dev_ctxs) {
  do {
    clearConsole();
    Table overalltable({ "Overview","" }, true);

    using namespace std::chrono;
    auto currenttime = high_resolution_clock::now();
    auto runningtime_ns = duration_cast<nanoseconds>(currenttime - starttime).count();
    const double runningtime = runningtime_ns / 1e9;
    overalltable.addRow({ "Running time: ", getFormattedDuration(runningtime) });

    uint64_t computed_hashes_total = 0;
    double currentspeed_total = 0;

    std::vector<Table> devicetables;
    for (uint32_t device_id = 0; device_id < dev_ctxs.size(); device_id++) {
      const DeviceContext& dev_ctx = dev_ctxs[device_id];

      if (dev_ctx.bestdifficulty > global_bestdifficulty) {
        global_bestdifficulty = dev_ctx.bestdifficulty;
        global_bestdifficulty_counter = dev_ctx.bestdifficulty_counter;
      }

      Table devtable({ "Device " + dev_ctx.device_name + "[" + std::to_string(device_id) + "]", "" }, true);


      devtable.addRow({ "Local/Global work size", std::to_string(dev_ctx.local_work_size) + "/" + std::to_string(dev_ctx.global_work_size) });


      const uint64_t computed_hashes_device = dev_ctx.completediterations_total;
      computed_hashes_total += computed_hashes_device;

      const double avgspeed_device = computed_hashes_device / runningtime;
      const double currentspeed_device = dev_ctx.getAvgSpeed();
      currentspeed_total += currentspeed_device;

      devtable.addRow({ "Current speed", getFormattedDouble(currentspeed_device) + "Hash/s" });
      devtable.addRow({ "Average speed", getFormattedDouble(avgspeed_device) + "Hash/s" });
      devtable.addRow({ "Scheduling", std::to_string(dev_ctx.completed_kernels / runningtime) + " Kernels/s" });

      devicetables.push_back(std::move(devtable));
    }

    const double unitspersecond_global = computed_hashes_total / runningtime;

    overalltable.addRow({ "Current speed [total]", getFormattedDouble(currentspeed_total) + "Hash/s" });
    overalltable.addRow({ "Average speed [total]", getFormattedDouble(unitspersecond_global) + "Hash/s" });


    const bool slowphase = TSUtil::isSlowPhase(identity.size(), startcounter);
    if (slowphase) {
      overalltable.addRow({ "Estimated time until slow phase", "0 (IN SLOW PHASE!!!)" });
    }
    else {
      auto its_until_slow = TSUtil::itsUntilSlowPhase(identity.size(), startcounter);
      auto time_until_slow = its_until_slow / currentspeed_total;
      overalltable.addRow({ "Estimated time until slow phase", getFormattedDuration(time_until_slow) });
    }

    uint64_t nextlevel = std::max((uint64_t)(global_bestdifficulty + 1ull),
      (uint64_t)MIN_TARGET_DIFFICULTY);

    // we want to estimate the remaining time for the next level,
    // accounting for the slow phase
    uint64_t nextlevel_estits = ((uint64_t)1 << nextlevel);
    uint64_t remaining_fastits = TSUtil::itsUntilSlowPhase(identity.size(), startcounter);
    uint64_t nextlevel_estits_adjusted = slowphase ? nextlevel_estits : (2 * nextlevel_estits - std::min(remaining_fastits, nextlevel_estits));
    double nextlevel_esttime_seconds = nextlevel_estits_adjusted / currentspeed_total;

    if (nextlevel_esttime_seconds > 60 && global_bestdifficulty + 1 < MIN_TARGET_DIFFICULTY) {
      MIN_TARGET_DIFFICULTY -= (uint8_t)std::ceil(std::log2(nextlevel_esttime_seconds / 60));
      MIN_TARGET_DIFFICULTY = std::max((uint8_t)32, MIN_TARGET_DIFFICULTY);
      // we need to recompute the time estimations
      nextlevel = std::max((uint64_t)(global_bestdifficulty + 1ull), (uint64_t)MIN_TARGET_DIFFICULTY);
      nextlevel_estits = ((uint64_t)1 << nextlevel);
      remaining_fastits = TSUtil::itsUntilSlowPhase(identity.size(), startcounter);
      nextlevel_estits_adjusted = slowphase ? nextlevel_estits : (2 * nextlevel_estits - std::min(remaining_fastits, nextlevel_estits));
      nextlevel_esttime_seconds = nextlevel_estits_adjusted / currentspeed_total;
    }
    overalltable.addRow({ "Security level" ,
      std::to_string((uint32_t)global_bestdifficulty) + " (with counter=" + std::to_string(global_bestdifficulty_counter) + ")" });

    overalltable.addRow({ "Estimated time until level " + std::to_string(nextlevel), getFormattedDuration(nextlevel_esttime_seconds) });
    overalltable.addRow({ "Current counter", std::to_string(startcounter) });


    // we start to print
    if (slowphase) {
      std::cout << "WARNING: You have entered the slow phase. With a fresh identity you could double your speed." << std::endl;
    }
    std::cout << overalltable.getTable();
    std::cout << std::endl << std::endl;
    for (auto& devtable : devicetables) {
      std::cout << devtable.getTable() << std::endl;
    }

    std::cout << std::endl << "(press Ctrl+C to stop and save progress)" << std::endl;
  } while (timerkiller.running() && timerkiller.wait_for(std::chrono::milliseconds(1000)));

  // we wait for all scheduled kernels to finish
  for (auto& dev : dev_ctxs) {
    dev.device->finish();
  }
  std::cout << std::endl << "===========================================================" << std::endl;
  std::cout << std::endl << "Stopped at counter: " << startcounter << std::endl;
}


void TSHasherContext::run_kernel_loop(DeviceContext* dev_ctx) {
  while (dev_ctx->tshasherctx->timerkiller.running()) {
    const size_t identity_length = dev_ctx->identitystring.size();
    bool use_kernel2 = TSUtil::isSlowPhase(identity_length, dev_ctx->tshasherctx->startcounter);

    dev_ctx->tshasherctx->startcounter_mutex.lock();
    auto dev_startcounter = dev_ctx->tshasherctx->startcounter;
    auto global_max_iterations = std::min((uint64_t)dev_ctx->global_work_size * KERNEL_STD_ITERATIONS, TSUtil::itsConstantCounterLength(dev_startcounter));
    const uint64_t iterations = global_max_iterations / dev_ctx->global_work_size;
    if (iterations == 0) {
      // if there are too few iterations until the counter length increases, we just skip these
      dev_ctx->tshasherctx->startcounter += global_max_iterations;
      dev_ctx->tshasherctx->startcounter_mutex.unlock();
      continue;
    }

    const uint8_t bestdifficulty = std::max(dev_ctx->tshasherctx->global_bestdifficulty, dev_ctx->bestdifficulty);
    const uint8_t targetdifficulty = std::max(dev_ctx->tshasherctx->MIN_TARGET_DIFFICULTY, (uint8_t)(1 + bestdifficulty));

    dev_ctx->measureTime();

    dev_ctx->lastscheduled_startcounter = dev_ctx->tshasherctx->startcounter;
    dev_ctx->tshasherctx->startcounter += static_cast<uint64_t>(dev_ctx->global_work_size) * iterations;
    dev_ctx->tshasherctx->startcounter_mutex.unlock();
    dev_ctx->schedulediterations_total += static_cast<uint64_t>(dev_ctx->global_work_size) * iterations;
    dev_ctx->lastschedulediterations_total = static_cast<uint64_t>(dev_ctx->global_work_size) * iterations;

    dev_ctx->device->setArgsAndLaunch(
      dev_startcounter,
      (uint32_t)iterations,
      targetdifficulty,
      (uint32_t)identity_length,
      dev_ctx->global_work_size,
      dev_ctx->local_work_size,
      use_kernel2);

    dev_ctx->device->finish();

    if (!dev_ctx->tshasherctx->timerkiller.running()) {
      return;
    }

    dev_ctx->device->readResults(dev_ctx->h_results, dev_ctx->global_work_size);
    read_kernel_result(dev_ctx);
  }
}

void TSHasherContext::read_kernel_result(DeviceContext* dev_ctx) {

  dev_ctx->completediterations_total += dev_ctx->lastschedulediterations_total;
  dev_ctx->completed_kernels++;

  // read the result
  uint8_t oldbestdifficulty = dev_ctx->bestdifficulty;
  for (uint32_t thread = 0; thread < dev_ctx->global_work_size; thread++) {
    if (dev_ctx->h_results[thread]) {
      // target found: now we search for the correct counter
      uint64_t its_per_worker = dev_ctx->lastschedulediterations_total / dev_ctx->global_work_size;
      uint64_t searchstartcounter = dev_ctx->lastscheduled_startcounter + its_per_worker * thread;
      uint8_t bestdifficulty = 0;
      uint64_t bestdifficulty_counter = 0;

      for (uint64_t counter = searchstartcounter;
        counter < searchstartcounter + its_per_worker;
        counter++) {
        uint8_t currentdifficulty = TSUtil::getDifficulty(dev_ctx->identitystring, counter);
        if (currentdifficulty > bestdifficulty) {
          bestdifficulty = currentdifficulty;
          bestdifficulty_counter = counter;
        }
      }

      if (bestdifficulty > dev_ctx->bestdifficulty) {
        dev_ctx->bestdifficulty = bestdifficulty;
        dev_ctx->bestdifficulty_counter = bestdifficulty_counter;
      }
      else if (bestdifficulty <= oldbestdifficulty) {
        // this should never happen
        std::cout << std::endl << "A critical error occurred." << std::endl;
        std::cout << "Claimed target could not be found." << std::endl;
        exit(-1);
      }
    }
  }
}
