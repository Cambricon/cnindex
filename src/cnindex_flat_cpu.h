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

#ifndef __CNINDEX_FLAT_CPU_H__
#define __CNINDEX_FLAT_CPU_H__

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "utils/thread_pool.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_flat_base.h"

namespace cnindex {

namespace impl {

class FlatCPU : public Flat {
 public:
  FlatCPU(int d, cnindexMetric_t metric);
  ~FlatCPU();

  cnindexReturn_t Reset() override;

  cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances,
                         bool output_on_mlu = false) const override;
  cnindexReturn_t Add(int n, const float *x, const int *ids) override;
  cnindexReturn_t Remove(int n, const int *ids) override;

  cnindexReturn_t GetData(float *x, int *ids) const override;
  const float * GetDataPointer() const override { return const_cast<const float *>(xb_.data()); };

  bool IsCPUImpl() const override { return true; };

 private:
  std::vector<float> xb_;
  std::vector<int> ids_;

#ifdef USE_THREAD_POOL
  int parallelism_ = 4;
  mutable EqualityThreadPool *thread_pool_ = nullptr;
  static std::atomic<int> instances_number_;
#endif

  mutable std::mutex mutex_;
  std::atomic<bool> exit_{false};
};  // FlatCPU

}  // impl

}  // cnindex

#endif  // __CNINDEX_FLAT_CPU_H__
