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

#include <algorithm>
#include <atomic>
#include <cmath>
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

#include "utils/arithmetics.h"
#include "utils/log.h"
#include "utils/thread_pool.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_flat_base.h"
#include "cnindex_ivf.h"
#include "cnindex_ivfpq_mlu200.h"

using cnindex::AllocMLUMemory;
using cnindex::FreeMLUMemory;
using cnindex::DeviceGuard;
using cnindex::CeilPower2;
using cnindex::GetCPUCoreNumber;
using cnindex::GetThreadPool;
using cnindex::EqualityThreadPool;
using cnindex::fvec_sub;

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU200

IVFPQ2::IVFPQ2(const cnindex::Flat *flat, cnindexMetric_t metric, int M, int nbits, int device_id)
    : IVFPQ(flat, metric, M, nbits, device_id) {
  LOGC(IVFPQ) << "\033[35mIVFPQ2::IVFPQ2()\033[0m";

  if (!(d_ == 256 || d_ == 512 || d_ == 768 || d_ == 1024) || !(M_ == 32 || M_ == 64) || nbits_ != 8) {
    LOGE(IVFPQ) << "IVFPQ2() bad parameters: d=" << d_ << ", M=" << M_ << ", nbits=" << nbits_;
    return;
  }

  CNRTInit();

#if CNRT_MAJOR_VERSION < 5
  cnrtDeviceInfo_t dev_info;
  cnrtGetDeviceInfo(&dev_info, device_id_);
  core_number_ = dev_info.core_num;
  op_limit_size_ = ((484 << 10) / 4 / vector_size_) & ~0xff;
#else
  int cluster_num, core_num_per_cluster;
  cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, device_id_);
  cnrtDeviceGetAttribute(&core_num_per_cluster, cnrtAttrMcorePerCluster, device_id_);
  core_number_ = cluster_num * core_num_per_cluster;
  int nram_size_per_core;
  cnrtDeviceGetAttribute(&nram_size_per_core, cnrtAttrNramSizePerMcore, device_id_);
  nram_size_per_core = 484 << 10;
  op_limit_size_ = (nram_size_per_core / 4 / vector_size_) & ~0xff;
#endif

  coarse_centroids_.resize(flat_->GetSize() * d_);
  flat_->GetData(coarse_centroids_.data());

  DeviceGuard(device_id_);

  int nlist = nlist_ == 1 ? core_number_ : nlist_;
  nlist_size_.resize(nlist, 0);
  nlist_bytes_.resize(nlist, 0);
  nlist_alloc_size_.resize(nlist, 0);
  vectors_ptr_.resize(nlist, nullptr);
  ids_ptr_.resize(nlist, nullptr);

  // alloc fixed memory
  size_t centroids_size = ALIGN_128(sizeof(float) * ksub_ * d_);
  size_t nlist_size = ALIGN_128(sizeof(int) * nlist);
  size_t ptrs_size = ALIGN_128(sizeof(void *) * nlist);
  size_t memory_size = centroids_size + 3 * nlist_size + 2 * ptrs_size;
  fixed_memory_mlu_ = AllocMLUMemory(memory_size);
  if (!fixed_memory_mlu_) return;
  cnrtMemset(fixed_memory_mlu_, 0, memory_size);
  centroids_mlu_ = fixed_memory_mlu_;
  nlist_size_mlu_ = static_cast<uint8_t *>(centroids_mlu_) + centroids_size;
  nlist_bytes_mlu_ = static_cast<uint8_t *>(nlist_size_mlu_) + nlist_size;
  nlist_alloc_size_mlu_ = static_cast<uint8_t *>(nlist_bytes_mlu_) + nlist_size;
  codes_ptr_mlu_ = static_cast<uint8_t *>(nlist_alloc_size_mlu_) + nlist_size;
  ids_ptr_mlu_ = static_cast<uint8_t *>(codes_ptr_mlu_) + ptrs_size;

  // create descriptors
  cnnlCreateTensorDescriptor(&centroids_desc_);
  cnnlCreateTensorDescriptor(&nlist_size_desc_);
  cnnlCreateTensorDescriptor(&codes_ids_ptr_desc_);
  const int dim[1] = { nlist };
  SetDesc(nlist_size_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);
  SetDesc(codes_ids_ptr_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);

  cnnlCreateTensorDescriptor(&nquery_desc_);
  cnnlCreateTensorDescriptor(&nprobe_indices_desc_);
  cnnlCreateTensorDescriptor(&topk_distances_desc_);
  cnnlCreateTensorDescriptor(&topk_ids_desc_);

  cnnlCreateTensorDescriptor(&add_residuals_desc_);
  cnnlCreateTensorDescriptor(&add_ids_desc_);
  cnnlCreateTensorDescriptor(&inserts_idx_desc_);
  cnnlCreateTensorDescriptor(&inserts_size_desc_);

  cnnlCreateTensorDescriptor(&remove_codes_desc_);
  cnnlCreateTensorDescriptor(&remove_ids_desc_);

  // cnrtSetDeviceFlag(CNRT_QUEUE_SYNC_YIELD);
  cnrtCreateQueue(&cnrt_queue_);
  cnnlCreate(&cnnl_handle_);
  cnnlSetQueue(cnnl_handle_, cnrt_queue_);

  thread_ = std::thread([this] { DeviceGuard(device_id_); while (TaskExecutor() >= 0); });
}

IVFPQ2::~IVFPQ2() {
  if (exit_) return;

  std::unique_lock<std::mutex> qlk(queue_mutex_);
  exit_ = true;
  qlk.unlock();

  queue_full_.notify_all();
  queue_empty_.notify_all();

  std::lock_guard<std::mutex> lk(mutex_);

  if (thread_.joinable()) thread_.join();

  DeviceGuard(device_id_);
  for (const auto &p : vectors_ptr_) FreeMLUMemory(p);
  FreeMLUMemory(fixed_memory_mlu_);
#ifndef USE_BFC_ALLOCATOR
  FreeMLUMemory(op_memory_mlu_);
#endif

  nlist_size_.clear();
  nlist_bytes_.clear();
  nlist_alloc_size_.clear();
  vectors_ptr_.clear();
  ids_ptr_.clear();

  if (centroids_desc_) cnnlDestroyTensorDescriptor(centroids_desc_);
  if (nlist_size_desc_) cnnlDestroyTensorDescriptor(nlist_size_desc_);
  if (codes_ids_ptr_desc_) cnnlDestroyTensorDescriptor(codes_ids_ptr_desc_);

  if (nquery_desc_) cnnlDestroyTensorDescriptor(nquery_desc_);
  if (nprobe_indices_desc_) cnnlDestroyTensorDescriptor(nprobe_indices_desc_);
  if (topk_distances_desc_) cnnlDestroyTensorDescriptor(topk_distances_desc_);
  if (topk_ids_desc_) cnnlDestroyTensorDescriptor(topk_ids_desc_);

  if (add_residuals_desc_) cnnlDestroyTensorDescriptor(add_residuals_desc_);
  if (add_ids_desc_) cnnlDestroyTensorDescriptor(add_ids_desc_);
  if (inserts_idx_desc_) cnnlDestroyTensorDescriptor(inserts_idx_desc_);
  if (inserts_size_desc_) cnnlDestroyTensorDescriptor(inserts_size_desc_);

  if (remove_codes_desc_) cnnlDestroyTensorDescriptor(remove_codes_desc_);
  if (remove_ids_desc_) cnnlDestroyTensorDescriptor(remove_ids_desc_);

  if (cnnl_handle_) cnnlDestroy(cnnl_handle_);
  if (cnrt_queue_) cnrtDestroyQueue(cnrt_queue_);

  LOGC(IVFPQ) << "\033[35mIVFPQ2::~IVFPQ2()\033[0m";
}

cnindexReturn_t IVFPQ2::Reset() {
  LOGT(IVFPQ) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  for (const auto &p : vectors_ptr_) FreeMLUMemory(p);

  int nlist = nlist_size_.size();
  nlist_size_.assign(nlist, 0);
  nlist_bytes_.assign(nlist, 0);
  nlist_alloc_size_.assign(nlist, 0);
  vectors_ptr_.assign(nlist, nullptr);
  ids_ptr_.assign(nlist, nullptr);

  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, vectors_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  ntotal_ = 0;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ2::SetCentroids(const float *centroids) {
  LOGT(IVFPQ) << "SetCentroids(" << static_cast<const void *>(centroids) << ")";

  if (!centroids) {
    LOGE(IVFPQ) << "SetCentroids() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  std::vector<float> centroids_trans(ksub_ * d_);
  // trans centroids: [M, ksub, dsub] -> [ksub, M, dsub]
  for (int i = 0; i < M_; i++) {
    for (int j = 0; j < ksub_; j++) {
      const float *src = centroids + (i * ksub_ + j) * dsub_;
      float *dst = centroids_trans.data() + (j * M_ + i) * dsub_;
      memcpy(dst, src, sizeof(float) * dsub_);
    }
  }
  { int d[3] = { ksub_, M_, dsub_ }; SetDesc(centroids_desc_, 3, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  cnrtMemcpy(centroids_mlu_, centroids_trans.data(), sizeof(float) * ksub_ * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ2::SetListData(int index, int size, const void *codes, const int *ids) {
  LOGT(IVFPQ) << "SetListData(" << index << ", " << size << ", " << static_cast<const void *>(codes)
              << ", " << static_cast<const void *>(ids) << ")";

  if (index >= nlist_ || index < 0 || !codes || !ids) {
    if (index >= nlist_ || index < 0) {
      LOGE(IVFPQ) << "SetListData() invalid list index: " << index;
    } else {
      LOGE(IVFPQ) << "SetListData() invalid parameters";
    }
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (size <= 0) {
    LOGW(IVFPQ) << "SetListData() list[" << index << "] is empty";
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int list_start = index;
  int list_end = index + 1;
  if (nlist_ == 1) {
    ntotal_ = size;
    size = ntotal_ / core_number_;
    list_end = ntotal_ % core_number_;
    for (int i = 0; i < core_number_; i++) {
      nlist_size_[i] = i < list_end ? (size + 1) : size;
    }
    list_start = 0;
    list_end = core_number_;
  } else {
    ntotal_ -= nlist_size_[index];
    ntotal_ += size;
    nlist_size_[index] = size;
  }

  int list_offset = 0;
  std::vector<uint8_t> codes_trans;
  for (int i = list_start; i < list_end; i++) {
    int list_size = nlist_size_[i];
    if (list_size <= 0) continue;
    nlist_bytes_[i] = list_size * vector_size_;
    // calculate allocate size.
    int alloc_size = CeilPower2(list_size);
    alloc_size = std::max(list_size, std::min(alloc_size, op_limit_size_));
    nlist_alloc_size_[i] = alloc_size;
    size_t codes_size = ALIGN_128(sizeof(uint8_t) * (size_t)alloc_size * vector_size_);
    size_t ids_size = ALIGN_128(sizeof(int) * (size_t)alloc_size);
    // allocate for codes and ids.
    FreeMLUMemory(vectors_ptr_[i]);
    vectors_ptr_[i] = AllocMLUMemory(codes_size + ids_size);
    if (!vectors_ptr_[i]) return CNINDEX_RET_ALLOC_FAILED;
    codes_trans.resize(nlist_bytes_[i]);
    const uint8_t *list_codes = static_cast<const uint8_t *>(codes) + list_offset * vector_size_;
    // trans codes: [list_size, code_size] -> [code_size, list_size].
    for (size_t i = 0; i < (size_t)list_size; i++) {
      for (int j = 0; j < vector_size_; j++) {
        codes_trans[j * (size_t)list_size + i] = list_codes[i * vector_size_ + j];
      }
    }
    cnrtMemcpy(vectors_ptr_[i], codes_trans.data(), nlist_bytes_[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
    ids_ptr_[i] = static_cast<uint8_t *>(vectors_ptr_[i]) + codes_size;
    if (!ids_ptr_[i]) return CNINDEX_RET_ALLOC_FAILED;
    int *list_ids = const_cast<int *>(ids) + list_offset;
    cnrtMemcpy(ids_ptr_[i], list_ids, sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_HOST2DEV);

    list_offset += list_size;
  }

  int nlist = nlist_size_.size();
  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, vectors_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

int IVFPQ2::GetListSize(int index) const {
  LOGT(IVFPQ) << "GetListSize(" << index << ")";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  if (index >= nlist_ || index < 0) {
    LOGE(IVFPQ) << "GetListSize() invalid list index: " << index;
    return CNINDEX_RET_BAD_PARAMS;
  }

  return nlist_ == 1 ? ntotal_ : nlist_size_[index];
}

cnindexReturn_t IVFPQ2::GetListData(int index, void *codes, int *ids) const {
  LOGT(IVFPQ) << "GetListData(" << index << ", " << static_cast<void *>(codes) << ", "
              << static_cast<void *>(ids) << ")";

  if (index >= nlist_ || index < 0) {
    LOGE(IVFPQ) << "GetListData() invalid list index: " << index;
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int list_start = nlist_ == 1 ? 0 : index;
  int list_end = nlist_ == 1 ? core_number_ : (index + 1);
  int offset = 0;

  std::vector<uint8_t> codes_trans;
  for (int i = list_start; i < list_end; i++) {
    int list_size = nlist_size_[i];
    if (list_size <= 0) {
      LOGD(IVFPQ) << "GetListData() list[" << index << "] is empty";
      continue;
    }
    if (codes) {
      if (!vectors_ptr_[i]) return CNINDEX_RET_NOT_VALID;
      int list_bytes = nlist_bytes_[i];
      uint8_t *list_codes = static_cast<uint8_t *>(codes) + (size_t)offset * vector_size_;
      codes_trans.resize(list_bytes);
      cnrtMemcpy(codes_trans.data(), vectors_ptr_[i], sizeof(uint8_t) * list_bytes, CNRT_MEM_TRANS_DIR_DEV2HOST);
      // trans codes: [code_size, list_size] -> [list_size, code_size].
      for (int i = 0; i < vector_size_; i++) {
        for (size_t j = 0; j < (size_t)list_size; j++) {
          list_codes[j * vector_size_ + i] = codes_trans[i * (size_t)list_size + j];
        }
      }
    }
    if (ids) {
      if (!ids_ptr_[i]) return CNINDEX_RET_NOT_VALID;
      cnrtMemcpy(ids + offset, ids_ptr_[i], sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
    offset += list_size;
  }

  return CNINDEX_RET_SUCCESS;
}

void IVFPQ2::ComputeResiduals(int n, const float *x, int nprobe, const int *indices, float *residuals) const {
  for (int i = 0; i < n; i++) {
    if (nlist_ == 1) {
      fvec_sub(x + i * d_, coarse_centroids_.data(), d_, residuals + i * nprobe * d_);
      for (int j = 1; j < nprobe; j++) {
        memcpy(residuals + (i * nprobe + j) * d_, residuals + i * nprobe * d_, sizeof(float) * d_);
      }
    } else {
      for (int j = 0; j < nprobe; j++) {
        int offset = i * nprobe + j;
        fvec_sub(x + i * d_, coarse_centroids_.data() + indices[offset] * d_, d_, residuals + offset * d_);
      }
    }
  }
}

cnindexReturn_t IVFPQ2::Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const {
  LOGT(IVFPQ) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << nprobe << ", " << k << ", "
              << static_cast<void *>(ids) << ", " << static_cast<void *>(distances) << ")";

  if (n <= 0 || !x || nprobe <= 0 || k <= 0 || !ids) {
    LOGE(IVFPQ) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (coarse_centroids_.empty() || !centroids_mlu_) {
    LOGE(IVFPQ) << "Search() centroids is empty";
    return CNINDEX_RET_NOT_VALID;
  }
  if (ntotal_ == 0) {
    LOGE(IVFPQ) << "Search() no vector";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int nlist = nlist_size_.size();
  if (nlist_ == 1) nprobe = nlist;
  nprobe = std::min(nprobe, nlist);

  size_t residuals_size = ALIGN_128(sizeof(float) * (size_t)n * nprobe * d_);
  size_t indices_size = ALIGN_128(sizeof(int) * (size_t)n * nprobe);
  size_t topk_distances_size = ALIGN_128(sizeof(float) * (size_t)n * k);
  size_t topk_ids_size = ALIGN_128(sizeof(int) * (size_t)n * k);
  size_t op_memory_size = residuals_size + indices_size + topk_distances_size + topk_ids_size;
  void *op_memory_mlu = nullptr;
#ifdef USE_BFC_ALLOCATOR
  op_memory_mlu = AllocMLUMemory(op_memory_size);
  if (!op_memory_mlu) return CNINDEX_RET_ALLOC_FAILED;
  unique_void_ptr_del op_memory_mlu_up(op_memory_mlu, [this](void *p) { FreeMLUMemory(p); });
#else
  if (op_memory_size_ < op_memory_size) {
    FreeMLUMemory(op_memory_mlu_);
    op_memory_mlu_ = AllocMLUMemory(op_memory_size);
    if (!op_memory_mlu_) return CNINDEX_RET_ALLOC_FAILED;
    op_memory_size_ = op_memory_size;
  }
  op_memory_mlu = op_memory_mlu_;
#endif
  void *residuals_mlu = op_memory_mlu;
  void *indices_mlu = static_cast<uint8_t *>(residuals_mlu) + residuals_size;
  void *topk_distances_mlu = static_cast<uint8_t *>(indices_mlu) + indices_size;
  void *topk_ids_mlu = static_cast<uint8_t *>(topk_distances_mlu) + topk_distances_size;

  int ndeal = 1;
  int64_t dealed = 0;
  std::vector<int> indices;
  std::vector<float> residuals(ndeal * nprobe * d_);
  std::promise<int> promise;
  int ret = CNINDEX_RET_SUCCESS;

  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;
    const float *x_t = x + dealed * d_;

    // prepare indices.
    if (nlist_ == 1) {  // prepare indices for nlist == 1.
      if (indices.empty()) {
        for (int i = 0; i < nprobe; i++) {
          indices.push_back(i);
        }
        for (int i = 1; i < ndeal; i++) {
          indices.insert(indices.end(), indices.begin(), indices.begin() + nprobe);
        }
      }
    } else {  // coarse search and reorder indices.
      if (indices.empty()) indices.resize(ndeal * nprobe);
      if (CNINDEX_RET_SUCCESS != flat_->Search(ndeal, x_t, nprobe, indices.data(), nullptr)) {
        LOGE(IVFPQ) << "Search() coarse search failed";
        break;
      }
      // reorder indices in descending order of list size.
      for (int i = 0; i < ndeal; i++) {
        std::sort(indices.begin() + i * nprobe, indices.begin() + (i + 1) * nprobe,
                  [this](int x, int y) -> bool { return nlist_size_[x] > nlist_size_[y]; });
      }
    }

    // check if topk use brute force search. indices is in descending order,
    // just compare k with minimum size of lists by all cores first batch processing.
    bool bfs = false;
    for (int i = 0; i < ndeal; i++) {
      if (nlist_size_[indices[i * nprobe + std::min(core_number_, nprobe) - 1]] < k) {
        bfs = true;
        break;
      }
    }

    // compute residuals
    ComputeResiduals(ndeal, x_t, nprobe, indices.data(), residuals.data());

    // copy to MLU
    float *residuals_t = static_cast<float *>(residuals_mlu) + dealed * nprobe * d_;
    int *indices_t = static_cast<int *>(indices_mlu) + dealed * nprobe;
    cnrtMemcpy(residuals_t, residuals.data(), sizeof(float) * ndeal * nprobe * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(indices_t, indices.data(), sizeof(int) * ndeal * nprobe, CNRT_MEM_TRANS_DIR_HOST2DEV);

    int *ids_t = static_cast<int *>(topk_ids_mlu) + dealed * k;
    float *distances_t = static_cast<float *>(topk_distances_mlu) + dealed * k;

    std::unique_lock<std::mutex> qlk(queue_mutex_);
    queue_full_.wait(qlk, [this] { return (exit_ || queue_.size() < queue_capacity_); } );
    if (exit_) return CNINDEX_RET_NOT_VALID;

    std::promise<int> *p_promise = (dealed + ndeal) >= n ? &promise : nullptr;
    Task task{ ndeal, residuals_t, nprobe, indices_t, k, ids_t, distances_t, bfs, p_promise };
    queue_.push(std::move(task));
    qlk.unlock();
    queue_empty_.notify_one();

    dealed += ndeal;
  }

  if (dealed >= n) {
    ret = promise.get_future().get();
    if (ret == CNINDEX_RET_SUCCESS) {
      cnrtMemcpy(ids, topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
      if (distances) cnrtMemcpy(distances, topk_distances_mlu, sizeof(float) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
  }

  return static_cast<cnindexReturn_t>(ret);
}

int IVFPQ2::TaskExecutor() const {
  Task task;
  std::unique_lock<std::mutex> qlk(queue_mutex_);
  queue_empty_.wait(qlk, [this] { return (exit_ || !queue_.empty()); });
  if (exit_) {
    while (!queue_.empty()) {
      task = queue_.front();
      queue_.pop();
      if (task.promise) task.promise->set_value(CNINDEX_RET_OP_FAILED);
    }
    cnrtSyncQueue(cnrt_queue_);
    return -1;
  }
  task = queue_.front();
  queue_.pop();
  qlk.unlock();
  queue_full_.notify_one();

  { int d[3] = { task.n, task.nprobe, d_ }; SetDesc(nquery_desc_, 3, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { task.n, task.nprobe }; SetDesc(nprobe_indices_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
  { int d[2] = { task.n, task.k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { task.n, task.k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

  cnnlStatus_t status = cnnlIvfProductQuantization(cnnl_handle_,
                                                   nquery_desc_, task.xr,
                                                   centroids_desc_, centroids_mlu_,
                                                   nlist_size_desc_, nlist_size_mlu_,
                                                   nlist_size_desc_, nlist_bytes_mlu_,
                                                   nlist_size_desc_, nlist_alloc_size_mlu_,
                                                   codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
                                                   codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
                                                   nprobe_indices_desc_, task.indices,
                                                   topk_distances_desc_, task.distances,
                                                   topk_ids_desc_, task.ids, task.bfs);

  if (status != CNNL_STATUS_SUCCESS) {
    LOGE(IVFPQ) << "TaskExecutor() invoke op failed";
  }
  if (task.promise) {
    cnrtSyncQueue(cnrt_queue_);
    task.promise->set_value(status == CNNL_STATUS_SUCCESS ? CNINDEX_RET_SUCCESS : CNINDEX_RET_OP_FAILED);
    return 1;
  }
  return 0;
}

cnindexReturn_t IVFPQ2::Add(int n, const float *x, const int *ids) {
  LOGT(IVFPQ) << "Add(" << n << ", " << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !x || !ids) {
    LOGE(IVFPQ) << "Add() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (coarse_centroids_.empty() || !centroids_mlu_) {
    LOGE(IVFPQ) << "Add() centroids is empty";
    return CNINDEX_RET_NOT_VALID;
  }
  if (((size_t)ntotal_ + n) > std::numeric_limits<int>::max()) {
    LOGE(IVFPQ) << "Add() vectors number to be added over int_max(" << std::numeric_limits<int>::max() << ")";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  using IntPair = std::pair<int, int>;
  int nlist = nlist_size_.size();
  int ndeal = 1024;
  int64_t dealed = 0;

  size_t residuals_size = ALIGN_128(sizeof(float) * ndeal * d_);
  size_t ids_size = ALIGN_128(sizeof(int) * ndeal);
  size_t codes_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t ids_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t list_idx_size = ALIGN_128(sizeof(int) * nlist);
  size_t insert_count_size = ALIGN_128(sizeof(int) * ndeal);
  size_t op_memory_size =
      residuals_size + ids_size + codes_ptr_size + ids_ptr_size + list_idx_size + insert_count_size;
  void *op_memory_mlu = nullptr;
#ifdef USE_BFC_ALLOCATOR
  op_memory_mlu = AllocMLUMemory(op_memory_size);
  if (!op_memory_mlu) return CNINDEX_RET_ALLOC_FAILED;
  unique_void_ptr_del op_memory_mlu_up(op_memory_mlu, [this](void *p) { FreeMLUMemory(p); });
#else
  if (op_memory_size_ < op_memory_size) {
    FreeMLUMemory(op_memory_mlu_);
    op_memory_mlu_ = AllocMLUMemory(op_memory_size);
    if (!op_memory_mlu_) return CNINDEX_RET_ALLOC_FAILED;
    op_memory_size_ = op_memory_size;
  }
  op_memory_mlu = op_memory_mlu_;
#endif
  void *residuals_mlu = op_memory_mlu;
  void *ids_mlu = static_cast<uint8_t *>(residuals_mlu) + residuals_size;
  void *codes_ptr_mlu = static_cast<uint8_t *>(ids_mlu) + ids_size;
  void *ids_ptr_mlu = static_cast<uint8_t *>(codes_ptr_mlu) + codes_ptr_size;
  void *list_idx_mlu = static_cast<uint8_t *>(ids_ptr_mlu) + ids_ptr_size;
  void *insert_size_mlu = static_cast<uint8_t *>(list_idx_mlu) + list_idx_size;

  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;
    const float *x_d = x + dealed * d_;

    std::vector<IntPair> inserts;  // [list_idx, insert_count]
    std::vector<int> indices;
    std::vector<float> x_vec;
    std::vector<int> ids_vec;

    if (nlist_ == 1) {  // try to balance list size
      std::vector<IntPair> complements;
      int ctotal = 0;
      int max = *std::max(nlist_size_.begin(), nlist_size_.end());
      for (int i = 0; i < nlist; i++) {
        int c = max - nlist_size_[i];
        complements.emplace_back(i, c);
        ctotal += c;
      }
      int ins_n = 0;
      int ins_r = 0;
      if (ndeal > ctotal) {
        int dr = ndeal - ctotal;
        ins_n = dr / nlist;
        ins_r = dr % nlist;
      }
      std::sort(complements.begin(), complements.end(),
                [](const IntPair &x, const IntPair &y) -> bool { return x.second > y.second; });
      ctotal = 0;
      for (int i = 0; i < nlist; i++) {
        int count = std::min(ndeal - ctotal, complements[i].second);
        count += i < ins_r ? (ins_n + 1) : ins_n;
        if (count > 0) inserts.emplace_back(complements[i].first, count);
        ctotal += count;
        if (ctotal >= ndeal) break;
      }

      indices.assign(ndeal, 0);
      ids_vec.assign(ids + dealed, ids + dealed + ndeal);
    } else {
      indices.resize(ndeal);
      std::vector<IntPair> idx_pairs;  // [x_idx, list_idx]
      if (CNINDEX_RET_SUCCESS != flat_->Search(ndeal, x_d, 1, indices.data(), nullptr)) {
        LOGE(IVFPQ) << "Add() flat search top 1 failed";
        break;
      }
      for (int i = 0; i < ndeal; i++) idx_pairs.emplace_back(i, indices[i]);

      // sort to gather vectors in same list.
      std::sort(idx_pairs.begin(), idx_pairs.end(),
                [](const IntPair &x, const IntPair &y) -> bool { return x.second < y.second; });

      int idx = -1, insert_count;
      for (int i = 0; i < ndeal; i++) {
        int x_idx, list_idx;
        std::tie(x_idx, list_idx) = idx_pairs[i];
        const float *xi = x + (dealed + x_idx) * d_;
        indices[i] = list_idx;
        x_vec.insert(x_vec.end(), xi, xi + d_);
        ids_vec.push_back(ids[dealed + x_idx]);
        if (idx != list_idx) {
          if (i > 0) inserts.emplace_back(idx, insert_count);
          idx = list_idx;
          insert_count = 1;
        } else {
          insert_count++;
        }
        if (i == (ndeal - 1)) inserts.emplace_back(idx, insert_count);
      }
      x_d = x_vec.data();
    }

    // compute residuals
    std::vector<float> residuals(ndeal * d_);
    ComputeResiduals(ndeal, x_d, 1, indices.data(), residuals.data());

    DeviceGuard(device_id_);

    // check if need reallocate memory
    std::vector<void *> codes_ptr(vectors_ptr_);
    std::vector<void *> ids_ptr(ids_ptr_);
    std::vector<int> free_list_idx;
    std::vector<int> inserts_idx, inserts_size;
    for (const auto &insert : inserts) {
      int list_idx, insert_count;
      std::tie(list_idx, insert_count) = insert;
      int new_size = nlist_size_[list_idx] + insert_count;
      if (new_size > std::min(nlist_alloc_size_[list_idx], op_limit_size_)) {
        // calculate allocate size.
        int alloc_size = CeilPower2(new_size);
        alloc_size = std::max(new_size, std::min(alloc_size, op_limit_size_));
        nlist_alloc_size_[list_idx] = alloc_size;
        size_t codes_size = ALIGN_128(vector_size_ * (size_t)alloc_size);
        size_t ids_size = ALIGN_128(sizeof(int) * (size_t)alloc_size);
        // allocate for codes and ids.
        codes_ptr[list_idx] = AllocMLUMemory(codes_size + ids_size);
        if (!codes_ptr[list_idx]) return CNINDEX_RET_ALLOC_FAILED;
        ids_ptr[list_idx] = static_cast<uint8_t *>(codes_ptr[list_idx]) + codes_size;
        free_list_idx.push_back(list_idx);
      }
      inserts_idx.push_back(list_idx);
      inserts_size.push_back(insert_count);
    }

    // copy to MLU
    int lists_idx = inserts_idx.size();
    int inserts_count = inserts_size.size();
    cnrtMemcpy(residuals_mlu, residuals.data(), sizeof(float) * ndeal * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_mlu, ids_vec.data(), sizeof(int) * ndeal, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(codes_ptr_mlu, codes_ptr.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_ptr_mlu, ids_ptr.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(list_idx_mlu, inserts_idx.data(), sizeof(int) * lists_idx, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(insert_size_mlu, inserts_size.data(), sizeof(int) * inserts_count, CNRT_MEM_TRANS_DIR_HOST2DEV);

    { int d[2] = { ndeal, d_ }; SetDesc(add_residuals_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[1] = { ndeal }; SetDesc(add_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { lists_idx }; SetDesc(inserts_idx_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { inserts_count }; SetDesc(inserts_size_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlIvfpqAdd(cnnl_handle_,
                                       add_residuals_desc_, residuals_mlu,
                                       add_ids_desc_, ids_mlu,
                                       centroids_desc_, centroids_mlu_,
                                       codes_ids_ptr_desc_, (void **)codes_ptr_mlu_,
                                       codes_ids_ptr_desc_, (void **)ids_ptr_mlu_,
                                       codes_ids_ptr_desc_, (void **)codes_ptr_mlu,
                                       codes_ids_ptr_desc_, (void **)ids_ptr_mlu,
                                       inserts_idx_desc_, list_idx_mlu,
                                       inserts_size_desc_, insert_size_mlu,
                                       nlist_size_desc_, nlist_size_mlu_);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(IVFPQ) << "Add() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtSyncQueue(cnrt_queue_);
      for (const auto &idx : free_list_idx) {
        FreeMLUMemory(vectors_ptr_[idx]);
        vectors_ptr_[idx] = codes_ptr[idx];
        ids_ptr_[idx] = ids_ptr[idx];
      }
      if (!free_list_idx.empty()) {
        cnrtMemcpy(codes_ptr_mlu_, vectors_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
        cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
      }
      ntotal_ += ndeal;
      cnrtMemcpy(nlist_size_.data(), nlist_size_mlu_, sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_DEV2HOST);
      LOGD(IVFPQ) << "Add() add " << ndeal << " vectors ok";
    }

    dealed += ndeal;
  }

  for (int i = 0; i < nlist; i++) nlist_bytes_[i] = vector_size_ * (size_t)nlist_size_[i];
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ2::Remove(int n, const int *ids) {
  LOGT(IVFPQ) << "Remove(" << n << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !ids) {
    LOGE(IVFPQ) << "Remove() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (ntotal_ == 0) {
    LOGW(IVFPQ) << "Remove() no vectors";
    return CNINDEX_RET_SUCCESS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

#ifdef USE_THREAD_POOL
  EqualityThreadPool *thread_pool = cnindex::GetThreadPool();
  size_t thread_pool_size = thread_pool->Size();
  int parallelism = 4;
  if (!flat_->flat_->IsCPUImpl()) {
    size_t threads_number = thread_pool_size + parallelism;
    threads_number = std::min(threads_number, GetCPUCoreNumber());
    if (threads_number > thread_pool_size) thread_pool->Resize(threads_number);
  }
#endif

  DeviceGuard(device_id_);

  int nlist = nlist_size_.size();
  for (int i = 0; i < n; i++) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    int id = ids[i];
    int list_idx = -1, offset = -1;

#ifdef USE_THREAD_POOL
    auto find_offset = [this](int index, int id) -> int {
      int list_size = nlist_size_[index];
      std::vector<int> code_ids(list_size);
      DeviceGuard(device_id_);
      cnrtMemcpy(code_ids.data(), ids_ptr_[index], sizeof(int) * list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
      auto it = std::find(code_ids.begin(), code_ids.end(), id);
      return (it != code_ids.end()) ? std::distance(code_ids.begin(), it) : -1;
    };

    int task_count = parallelism;
    int idx = 0;
    std::vector<std::future<int>> fs;
    for (int j = 0; j < nlist; j++) {
      if (task_count == parallelism) idx = j;
      fs.emplace_back(thread_pool->Push(find_offset, j, id));
      if (--task_count == 0 || j == (nlist - 1)) {
        for (auto &f : fs) {
          offset = f.get();
          if (offset != -1) {
            list_idx = idx + std::distance(fs.data(), &f);
            break;
          }
        }
        task_count = parallelism;
        fs.clear();
        if (list_idx >= 0 && offset >= 0) break;
      }
    }
#else
    for (int j = 0; j < nlist; j++) {
      int list_size = nlist_size_[j];
      std::vector<int> code_ids(list_size);
      cnrtMemcpy(code_ids.data(), ids_ptr_[j], sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
      auto it = std::find(code_ids.begin(), code_ids.end(), id);
      if (it != code_ids.end()) {
        list_idx = j;
        offset = std::distance(code_ids.begin(), it);
        break;
      }
    }
#endif
    if (list_idx == -1 || offset == -1) {
      LOGE(IVFPQ) << "Remove() find id: " <<  id << " failed";
#ifdef USE_THREAD_POOL
      if (!flat_->flat_->IsCPUImpl()) thread_pool->Resize(thread_pool_size);
#endif
      return CNINDEX_RET_NOT_VALID;
    }

    LOGD(IVFPQ) << "Remove() find id[" << i << "]: " << id << " in list: " << list_idx << " offset: " << offset;

    int list_size = nlist_size_[list_idx];
    { int d[2] = { vector_size_, list_size }; SetDesc(remove_codes_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_UINT8); }
    { int d[1] = { list_size }; SetDesc(remove_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlIvfpqRemove(cnnl_handle_,
                                          remove_codes_desc_, (uint8_t *)vectors_ptr_[list_idx],
                                          remove_ids_desc_, (int *)ids_ptr_[list_idx],
                                          nlist_size_desc_, (int *)nlist_size_mlu_,
                                          list_idx, offset);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(IVFPQ) << "Remove() invoke op failed";
#ifdef USE_THREAD_POOL
      if (!flat_->flat_->IsCPUImpl()) thread_pool->Resize(thread_pool_size);
#endif
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtSyncQueue(cnrt_queue_);
      LOGD(IVFPQ) << "Remove() remove id: " << id << " ok";
      ntotal_--;
      nlist_size_[list_idx]--;
    }
  }
#ifdef USE_THREAD_POOL
  if (!flat_->flat_->IsCPUImpl()) thread_pool->Resize(thread_pool_size);
#endif
  return CNINDEX_RET_SUCCESS;
}

#endif  // ENABLE_MLU200

}  // impl

}  // cnindex
