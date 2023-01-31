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
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/bfc_allocator_test.cc
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

#include <algorithm>
#include <random>

#include "../src/utils/bfc_allocator.h"

// A fake SubAllocator to test the performance of BFCAllocator.
class FakeSubAllocator : public cnindex::SubAllocator {
 public:
  FakeSubAllocator() : cnindex::SubAllocator({}, {}), alloc_counter_(0) {}
  ~FakeSubAllocator() override {}

  // Alloc and Free functions are implemented as very cheap operations, so that
  // the benchmark can focus on the performance of BFCAllocator itself.
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    *bytes_received = num_bytes;
    return reinterpret_cast<void*>(alloc_counter_++);
  }

  void Free(void* ptr, size_t num_bytes) override {}

  bool SupportsCoalescing() const override { return false; }

 private:
  int64_t alloc_counter_;
};

int main(int argc, char* argv[]) {
  constexpr int kAllocSize = 1 << 14;
  const int kLongLivedObjects = 200;
  const int kShortLivedObjects = 200;

  cnindex::BFCAllocator bfc_allocator(new FakeSubAllocator, 1 << 30, false, "test");

  std::string test_op_name = "test_op";
  cnindex::ScopedMemoryDebugAnnotation annotation(test_op_name.data());

  // Allocate long lived objects.
  std::vector<void*> long_lived(kLongLivedObjects);
  for (int i = 0; i < kLongLivedObjects; i++) {
    long_lived[i] = bfc_allocator.AllocateRaw(1, kAllocSize);
  }
  std::vector<int> deallocation_order(kShortLivedObjects);
  for (int i = 0; i < kShortLivedObjects; i++) {
    deallocation_order[i] = i;
  }
  std::shuffle(deallocation_order.begin(), deallocation_order.end(),
               std::default_random_engine(0));

  // Allocate and deallocate short lived objects.
  std::vector<void*> short_lived(kShortLivedObjects);
  for (int i = 0; i < kShortLivedObjects; i++) {
    short_lived[i] = bfc_allocator.AllocateRaw(1, kAllocSize);
  }
  for (int i = 0; i < kShortLivedObjects; i++) {
    bfc_allocator.DeallocateRaw(short_lived[deallocation_order[i]]);
  }
}
