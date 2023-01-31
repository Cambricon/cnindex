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

#ifndef __CNINDEX_FLAT_MLU300_H__
#define __CNINDEX_FLAT_MLU300_H__

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include <cnrt.h>
#include <cnnl.h>
#include <cnnl_extra.h>

#include "cnindex.h"
#include "cnindex_flat_base.h"

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU300

class Flat3 : public Flat {
 public:
  Flat3(int d, cnindexMetric_t metric, int device_id);
  ~Flat3();

  cnindexReturn_t Reset() override;

  cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances,
                         bool output_on_mlu = false) const override;
  cnindexReturn_t Add(int n, const float *x, const int *ids) override;
  cnindexReturn_t Remove(int n, const int *ids) override;

  cnindexReturn_t GetData(float *x, int *ids) const override;
  const float * GetDataPointer() const override {
    if (ntotal_ == 0) return nullptr;
    return const_cast<const float *>(static_cast<float *>(vectors_base_mlu_));
  }

  bool IsCPUImpl() const override { return false; };
 private:
  int GetMaxBatch(int batch, int dimension, int k, int loop_deal_nlib) const;
  using unique_void_ptr_del = std::unique_ptr<void, std::function<void(void *)>>;

#define SetDesc(desc, n, d, l, t) cnnlSetTensorDescriptor(desc, l, t, n, d)

  int nallocated_;
  void *vectors_base_mlu_ = nullptr;
  std::vector<int> ids_;

#ifndef USE_BFC_ALLOCATOR
  mutable void *op_memory_mlu_ = nullptr;
  mutable int op_memory_size_ = 0;
#endif

  cnnlFlatSearchStruct_t search_desc_ = nullptr;
  cnnlTensorDescriptor_t vectors_base_desc_ = nullptr;
  cnnlTensorDescriptor_t query_vectors_desc_ = nullptr;
  cnnlTensorDescriptor_t topk_distances_desc_ = nullptr;
  cnnlTensorDescriptor_t topk_ids_desc_ = nullptr;

  const int device_id_;
  int core_number_;
  int cluster_num_;
  cnrtQueue_t cnrt_queue_ = nullptr;
  cnnlHandle_t cnnl_handle_ = nullptr;

  mutable std::mutex mutex_;
  std::atomic<bool> exit_{false};
};  // Flat3

#endif  // ENABLE_MLU300

}  // impl

}  // cnindex

#endif  // __CNINDEX_FLAT_MLU300_H__
