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

#ifndef __CNINDEX_IVFPQ_MLU200_H__
#define __CNINDEX_IVFPQ_MLU200_H__

#include <atomic>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <cnrt.h>
#include <cnnl.h>
#include <cnnl_extra.h>

#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_ivfpq_base.h"

using cnindex::EqualityThreadPool;

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU200

class IVFPQ2 : public IVFPQ {
 public:
  IVFPQ2(const cnindex::Flat *flat, cnindexMetric_t metric, int M, int nbits, int device_id);
  ~IVFPQ2();

  cnindexReturn_t Reset() override;

  cnindexReturn_t SetCentroids(const float *centroids) override;

  cnindexReturn_t SetListData(int index, int size, const void *vectors, const int *ids) override;
  int GetListSize(int index) const override;
  cnindexReturn_t GetListData(int index, void *vectors, int *ids) const override;

  cnindexReturn_t Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const override;
  cnindexReturn_t Add(int n, const float *x, const int *ids) override;
  cnindexReturn_t Remove(int n, const int *ids) override;

 private:
  using unique_void_ptr_del = std::unique_ptr<void, std::function<void(void *)>>;

#define SetDesc(desc, n, d, l, t) cnnlSetTensorDescriptor(desc, l, t, n, d)

  void ComputeResiduals(int n, const float *x, int nprobe, const int *indices, float *residuals) const;
  int TaskExecutor() const;

  std::vector<float> coarse_centroids_;

  std::vector<int> nlist_bytes_;
  std::vector<int> nlist_alloc_size_;

  void *fixed_memory_mlu_ = nullptr;
  void *centroids_mlu_ = nullptr;
  void *nlist_size_mlu_ = nullptr;
  void *nlist_bytes_mlu_ = nullptr;
  void *nlist_alloc_size_mlu_ = nullptr;
  void *codes_ptr_mlu_ = nullptr;
  void *ids_ptr_mlu_ = nullptr;
#ifndef USE_BFC_ALLOCATOR
  mutable void *op_memory_mlu_ = nullptr;
  mutable int op_memory_size_ = 0;
#endif

  // common descriptors
  cnnlTensorDescriptor_t centroids_desc_ = nullptr;
  cnnlTensorDescriptor_t nlist_size_desc_ = nullptr;
  cnnlTensorDescriptor_t codes_ids_ptr_desc_ = nullptr;

  // search descriptors
  cnnlTensorDescriptor_t nquery_desc_ = nullptr;
  cnnlTensorDescriptor_t nprobe_indices_desc_ = nullptr;
  cnnlTensorDescriptor_t topk_distances_desc_ = nullptr;
  cnnlTensorDescriptor_t topk_ids_desc_ = nullptr;

  // add descriptors
  cnnlTensorDescriptor_t add_residuals_desc_ = nullptr;
  cnnlTensorDescriptor_t add_ids_desc_ = nullptr;
  cnnlTensorDescriptor_t inserts_idx_desc_ = nullptr;
  cnnlTensorDescriptor_t inserts_size_desc_ = nullptr;

  // remove descriptors
  cnnlTensorDescriptor_t remove_codes_desc_ = nullptr;
  cnnlTensorDescriptor_t remove_ids_desc_ = nullptr;

  cnrtQueue_t cnrt_queue_ = nullptr;
  cnnlHandle_t cnnl_handle_ = nullptr;
  int core_number_;
  int op_limit_size_;  // MLU operator data loading limitation

  struct Task {
    int n;
    float *xr;
    int nprobe;
    int *indices;
    int k;
    int *ids;
    float *distances;
    bool bfs;  // brute force search
    std::promise<int> *promise;
  };

  mutable std::mutex queue_mutex_;
  mutable std::condition_variable queue_full_;
  mutable std::condition_variable queue_empty_;
  mutable std::queue<Task> queue_;
  const size_t queue_capacity_ = 32;
  std::thread thread_;

  mutable std::mutex mutex_;
  std::atomic<bool> exit_{false};
};  // IVFPQ2

#else

class IVFPQ2 : public IVFPQ {};

#endif  // ENABLE_MLU200

}  // impl

}  // cnindex

#endif  // __CNINDEX_IVFPQ_MLU200_H__
