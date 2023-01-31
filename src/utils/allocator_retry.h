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
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/allocator_retry.h
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

#ifndef __CNINDEX_ALLOCATOR_RETRY_H__
#define __CNINDEX_ALLOCATOR_RETRY_H__

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>

namespace cnindex {

// A retrying wrapper for a memory allocator.
class AllocatorRetry {
 public:
  AllocatorRetry();

  // Call 'alloc_func' to obtain memory.  On first call,
  // 'verbose_failure' will be false.  If return value is nullptr,
  // then wait up to 'max_millis_to_wait' milliseconds, retrying each
  // time a call to DeallocateRaw() is detected, until either a good
  // pointer is returned or the deadline is exhausted.  If the
  // deadline is exhausted, try one more time with 'verbose_failure'
  // set to true.  The value returned is either the first good pointer
  // obtained from 'alloc_func' or nullptr.
  void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes,
                                        bool verbose_failure)>
                        alloc_func,
                    int max_millis_to_wait, size_t alignment, size_t bytes);

  // Called to notify clients that some memory was returned.
  void NotifyDealloc();

 private:
  // Env* env_;
  std::mutex mu_;
  std::condition_variable memory_returned_;
};

// Implementation details below
inline void AllocatorRetry::NotifyDealloc() {
  // std::lock_guard<std::mutex> l(mu_);
  memory_returned_.notify_all();
}

}  // namespace cnindex

#endif  // __CNINDEX_ALLOCATOR_RETRY_H__
