/*************************************************************************
* Copyright (C) 2021 by Cambricon, Inc. All rights reserved
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* A part of this source code is referenced from google tensorflow project.
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/allocator_retry.cc
* Copyright 2015 The TensorFlow Authors. 
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

#include <chrono>

#include "allocator_retry.h"

namespace cnindex {

static inline uint64_t Now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

namespace {
class ScopedTimeTracker {
 public:
  explicit ScopedTimeTracker() {}
  void Enable() {
    if (!started) {  // Only override start_us when not set yet.
      start_us_ = Now();
      started = true;
    }
  }
  ~ScopedTimeTracker() {
    if (started) {
      uint64_t end_us = Now();
      // metrics::UpdateBfcAllocatorDelayTime(end_us - *start_us_);
    }
  }

 private:
  uint64_t start_us_;
  bool started = false;
};
}  // namespace

AllocatorRetry::AllocatorRetry() {}

void* AllocatorRetry::AllocateRaw(
    std::function<void*(size_t alignment, size_t num_bytes,
                        bool verbose_failure)>
        alloc_func,
    int max_millis_to_wait, size_t alignment, size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  ScopedTimeTracker tracker;
  uint64_t deadline_micros = 0;
  bool first = true;
  void* ptr = nullptr;
  while (ptr == nullptr) {
    ptr = alloc_func(alignment, num_bytes, false);
    if (ptr == nullptr) {
      uint64_t now = Now();
      if (first) {
        deadline_micros = now + max_millis_to_wait * 1000;
        first = false;
      }
      if (now < deadline_micros) {
        tracker.Enable();
        std::unique_lock<std::mutex> l(mu_);
        memory_returned_.wait_for(l,
            std::chrono::milliseconds((deadline_micros - now) / 1000));
      } else {
        return alloc_func(alignment, num_bytes, true);
      }
    }
  }
  return ptr;
}

}  // namespace cnindex
