/*************************************************************************
* Copyright (C) 2021 by Cambricon, Inc. All rights reserved
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*************************************************************************/

#ifdef _MSC_VER
#include "windows.h"
#else
#include "sys/sysinfo.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include <cnrt.h>

#include "bfc_allocator.h"
#include "log.h"
#include "thread_pool.h"
#include "utils.h"

namespace cnindex {

void CNRTInit() {
#if CNRT_MAJOR_VERSION < 5
  static std::once_flag of;
  std::call_once(of, []() { cnrtInit(0); });
#endif
}

void DeviceGuard(int device_id) {
  if (device_id < 0) return;
#if CNRT_MAJOR_VERSION < 5
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, device_id);
  cnrtSetCurrentDevice(dev);
#else
  cnrtSetDevice(device_id);
#endif
}

#ifdef USE_BFC_ALLOCATOR
#define RESERVED_MEMORY_SIZE 500  // reserved memory size for others in unit of Mega bytes
// SubAllocator for CNRT
class CNRTAllocator : public SubAllocator {
 public:
  CNRTAllocator() : SubAllocator({}, {}) {}
  ~CNRTAllocator() override {}

  void * Alloc(size_t alignment, size_t num_bytes, size_t *bytes_received) override {
    void *ptr = nullptr;
    if (CNRT_RET_SUCCESS == cnrtMalloc(&ptr, num_bytes)) {
      *bytes_received = num_bytes;
    } else {
      LOGE(CNRTAllocator) << "CNRTAllocator() cnrtMalloc(" << num_bytes << ") failed";
      *bytes_received = 0;
      ptr = nullptr;
    }
    return ptr;
  }
  void Free(void *ptr, size_t num_bytes) override {
    if (CNRT_RET_SUCCESS != cnrtFree(ptr)) {
      LOGE(CNRTAllocator) << "CNRTAllocator() cnrtFree(" << ptr << ") failed";
    }
  }
  bool SupportsCoalescing() const override { return false; }
};

static inline std::shared_ptr<BFCAllocator> GetAllocator(int device_id = -1) {
  static const size_t devices_max_num = 64;
  static std::mutex alloc_mutex;
  static std::vector<std::shared_ptr<BFCAllocator>> allocators(devices_max_num, nullptr);

  std::lock_guard<std::mutex> lk(alloc_mutex);
  unsigned int dev_ordinal;
  if (device_id < 0) {
#if CNRT_MAJOR_VERSION <= 4
#if CNRT_MINOR_VERSION <= 8
    LOGE(CNRTAllocator) << "GetAllocator() need device id for cnrt version <= 4.8";
    return nullptr;
#else
    if (CNRT_RET_SUCCESS != cnrtGetCurrentOrdinal(&dev_ordinal)) {
      LOGE(CNRTAllocator) << "GetAllocator() cnrtGetCurrentOrdinal failed";
      return nullptr;
    }
#endif
#else
    if (CNRT_RET_SUCCESS != cnrtGetDevice(reinterpret_cast<int *>(&dev_ordinal))) {
      LOGE(CNRTAllocator) << "GetAllocator() cnrtGetDevice failed";
      return nullptr;
    }
#endif
    if (dev_ordinal >= (devices_max_num - 1)) {
      LOGE(CNRTAllocator) << "GetAllocator() dev_ordinal >= (devices_max_num - 1)[" << (devices_max_num - 1) << "]";
      return nullptr;
    }
  } else {
    dev_ordinal = device_id;
  }
  if (!allocators[dev_ordinal]) {
    size_t free_mem, total_mem;
#if CNRT_MAJOR_VERSION < 5
    cnrtGetMemInfo(&free_mem, &total_mem, CNRT_CHANNEL_TYPE_DUPLICATE);
#else
    cnrtMemGetInfo(&free_mem, &total_mem);
    free_mem >>= 20;
#endif
    LOGD(IVFPQ) << "GetAllocator() free memory: " << free_mem << "M bytes on device " << dev_ordinal;
    size_t bfc_mem_size = free_mem - RESERVED_MEMORY_SIZE;
    allocators[dev_ordinal].reset(new BFCAllocator(new CNRTAllocator, bfc_mem_size << 20, true, "cnindex"));
  }
  return allocators[dev_ordinal];
}
#endif

void * AllocMLUMemory(size_t bytes, int device_id) {
#ifdef USE_BFC_ALLOCATOR
  auto allocator = GetAllocator(device_id);
  return allocator ? allocator->AllocateRaw(128, bytes) : nullptr;
#else
  void *ptr = nullptr;
  if (CNRT_RET_SUCCESS != cnrtMalloc(&ptr, bytes)) {
    LOGE(IVFPQ) << "AllocMLUMemory() cnrtMalloc(" << bytes << ") failed";
    ptr = nullptr;
  }
  return ptr;
#endif
}

void FreeMLUMemory(void *ptr, int device_id) {
  if (!ptr) return;
#ifdef USE_BFC_ALLOCATOR
  auto allocator = GetAllocator(device_id);
  if (allocator) allocator->DeallocateRaw(ptr);
#else
  if (CNRT_RET_SUCCESS != cnrtFree(ptr)) {
    LOGE(IVFPQ) << "AllocMLUMemory() cnrtFree(" << ptr << ") failed";
  }
#endif
}

static inline int ifo(uint64_t n) {
#if defined(__GNUC__)
  return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS) && (_WIN64)
  unsigned long index;
  _BitScanReverse64(&index, n);
  return index;
#else
  int r = 0;
  while (n > 0) {
    r++;
    n >>= 1;
  }
  return (r - 1);
#endif
}

uint32_t CeilPower2(uint32_t n) {
  if (n == 0) return 0;
  n = n > 1 ? (n - 1) : n;
  return (2 << ifo(n));
}

uint32_t FloorPower2(uint32_t n) {
  if (n <= 1) return 0;
  return (1 << ifo(n));
}

size_t GetCPUCoreNumber() {
#ifdef _MSC_VER
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
#else
  /* return sysconf(_SC_NPROCESSORS_ONLN); */

  // GNU way
  return get_nprocs();
#endif
}

#ifdef USE_THREAD_POOL
static std::mutex tp_mutex;
static std::unique_ptr<EqualityThreadPool> thread_pool{nullptr};

EqualityThreadPool * GetThreadPool() {
  std::lock_guard<std::mutex> lk(tp_mutex);
  if (!thread_pool) thread_pool.reset(new EqualityThreadPool(nullptr));
  return thread_pool.get();
}
#endif

}  // namespace cnindex
