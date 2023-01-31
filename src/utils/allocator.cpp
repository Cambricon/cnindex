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
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.cc
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

#include <atomic>

#include "allocator.h"

namespace cnindex {

thread_local MemoryDebugAnnotation ScopedMemoryDebugAnnotation::annotation_;

std::string AllocatorStats::DebugString() const {
  char buf[1024] = {0};
  snprintf(buf, 1024,
      "Limit:            %20lld\n"
      "InUse:            %20lld\n"
      "MaxInUse:         %20lld\n"
      "NumAllocs:        %20lld\n"
      "MaxAllocSize:     %20lld\n"
      "Reserved:         %20lld\n"
      "PeakReserved:     %20lld\n"
      "LargestFreeBlock: %20lld\n",
      static_cast<long long>(this->bytes_limit),
      static_cast<long long>(this->bytes_in_use),
      static_cast<long long>(this->peak_bytes_in_use),
      static_cast<long long>(this->num_allocs),
      static_cast<long long>(this->largest_alloc_size),
      static_cast<long long>(this->bytes_reserved),
      static_cast<long long>(this->peak_bytes_reserved),
      static_cast<long long>(this->largest_free_block_bytes));
  return std::string(buf);
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {}
/*
// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

void EnableCPUAllocatorFullStats() { cpu_allocator_collect_full_stats = true; }
bool CPUAllocatorFullStatsEnabled() { return cpu_allocator_collect_full_stats; }
*/
std::string AllocatorAttributes::DebugString() const {
  return ("AllocatorAttributes(on_host=" + std::to_string(on_host()) +
          " nic_compatible=" + std::to_string(nic_compatible()) +
          " gpu_compatible=" + std::to_string(gpu_compatible()) + ")");
}
/*
Allocator* cpu_allocator_base() {
  static Allocator* cpu_alloc =
      AllocatorFactoryRegistry::singleton()->GetAllocator();
  // TODO(tucker): This really seems wrong.  It's only going to be effective on
  // the first call in a process (but the desired effect is associated with a
  // session), and we probably ought to be tracking the highest level Allocator,
  // not the lowest.  Revisit the advertised semantics of the triggering option.
  if (cpu_allocator_collect_full_stats && !cpu_alloc->TracksAllocationSizes()) {
    cpu_alloc = new TrackingAllocator(cpu_alloc, true);
  }
  return cpu_alloc;
}

Allocator* cpu_allocator(int numa_node) {
  // Correctness relies on devices being created prior to the first call
  // to cpu_allocator, if devices are ever to be created in the process.
  // Device creation in turn triggers ProcessState creation and the availability
  // of the correct access pointer via this function call.
  static ProcessStateInterface* ps =
      AllocatorFactoryRegistry::singleton()->process_state();
  if (ps) {
    return ps->GetCPUAllocator(numa_node);
  } else {
    return cpu_allocator_base();
  }
}
*/
SubAllocator::SubAllocator(const std::vector<Visitor>& alloc_visitors,
                           const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors) {}

void SubAllocator::VisitAlloc(void* ptr, int index, size_t num_bytes) {
  for (const auto& v : alloc_visitors_) {
    v(ptr, index, num_bytes);
  }
}

void SubAllocator::VisitFree(void* ptr, int index, size_t num_bytes) {
  // Although we don't guarantee any order of visitor application, strive
  // to apply free visitors in reverse order of alloc visitors.
  for (int i = free_visitors_.size() - 1; i >= 0; --i) {
    free_visitors_[i](ptr, index, num_bytes);
  }
}
}  // namespace cnindex
