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

#include "utils/log.h"
#include "utils/thread_pool.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_flat_base.h"
#include "cnindex_ivf.h"
#include "cnindex_ivfpq.h"
#include "cnindex_ivfpq_mlu300.h"

using cnindex::AllocMLUMemory;
using cnindex::FreeMLUMemory;
using cnindex::DeviceGuard;
using cnindex::CeilPower2;
using cnindex::GetCPUCoreNumber;
using cnindex::GetThreadPool;
using cnindex::EqualityThreadPool;

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU300

IVFPQ3::IVFPQ3(const cnindex::Flat *flat, cnindexMetric_t metric, int M, int nbits, int device_id)
    : IVFPQ(flat, metric, M, nbits, device_id) {
  bool pipeline_mode = flat->flat_->IsCPUImpl() && pipeline_mode_;
  LOGC(IVFPQ) << "\033[35mIVFPQ3::IVFPQ3(" << (pipeline_mode ? "P" : "") << ")\033[0m";

  if (!(d_ == 256 || d_ == 512 || d_ == 768 || d_ == 1024) || !(M_ == 32 || M_ == 64) || nbits_ != 8) {
    LOGE(IVFPQ) << "IVFPQ3() bad parameters: d=" << d_ << ", M=" << M_ << ", nbits=" << nbits_;
    return;
  }

  int cluster_num, core_num_per_cluster;
  cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, device_id_);
  cnrtDeviceGetAttribute(&core_num_per_cluster, cnrtAttrMcorePerCluster, device_id_);
  core_number_ = cluster_num * core_num_per_cluster;
  int nram_size_per_core;
  cnrtDeviceGetAttribute(&nram_size_per_core, cnrtAttrNramSizePerMcore, device_id_);
  int reserved_nram_size = 128 << 10;
  int nram_size = nram_size_per_core - reserved_nram_size - 4 * nlist_ * 8 - nlist_;
  int m_align = (M_ / 64 + (int)(M_ % 64 > 0)) * 64;
  op_limit_size_ = (nram_size_per_core / m_align / vector_size_) & ~0xff;

  DeviceGuard(device_id_);

  int nlist = nlist_;
  nlist_size_.resize(nlist, 0);
  nlist_alloc_size_.resize(nlist, 0);
  vectors_ptr_.resize(nlist, nullptr);
  ids_ptr_.resize(nlist, nullptr);

  // alloc fixed memory
  size_t coarse_centroids_size = ALIGN_128(sizeof(float) * flat_->GetSize() * d_);
  size_t pq_centroids_size = ALIGN_128(sizeof(float) * ksub_ * d_);
  size_t nlist_size = ALIGN_128(sizeof(int) * nlist);
  size_t ptrs_size = ALIGN_128(sizeof(void *) * nlist);
  size_t memory_size = flat_->flat_->IsCPUImpl() ? coarse_centroids_size : 0;
  memory_size += pq_centroids_size + nlist_size + 2 * ptrs_size;
 
  fixed_memory_mlu_ = AllocMLUMemory(memory_size);
  if (!fixed_memory_mlu_) return;
  cnrtMemset(fixed_memory_mlu_, 0, memory_size);

  if (flat_->flat_->IsCPUImpl()) {
    coarse_centroids_mlu_ = fixed_memory_mlu_;
    const float *coarse_centroids = flat_->flat_->GetDataPointer();
    if (!coarse_centroids) {
      LOGE(IVFPQ) << "IVFPQ3() get coarse centroids on CPU failed";
      return;
    }
    cnrtMemcpy(coarse_centroids_mlu_, const_cast<float *>(coarse_centroids), sizeof(float) * flat_->GetSize() * d_,
               CNRT_MEM_TRANS_DIR_HOST2DEV);
    pq_centroids_mlu_ = static_cast<uint8_t *>(coarse_centroids_mlu_) + coarse_centroids_size;
  } else {
    coarse_centroids_mlu_ = const_cast<float *>(flat_->flat_->GetDataPointer());
    if (!coarse_centroids_mlu_) {
      LOGE(IVFPQ) << "IVFPQ3() get coarse centroids on MLU failed";
      return;
    }
    pq_centroids_mlu_ = fixed_memory_mlu_;
  }
  nlist_size_mlu_ = static_cast<uint8_t *>(pq_centroids_mlu_) + pq_centroids_size;
  codes_ptr_mlu_ = static_cast<uint8_t *>(nlist_size_mlu_) + nlist_size;
  ids_ptr_mlu_ = static_cast<uint8_t *>(codes_ptr_mlu_) + ptrs_size;

  // create descriptors
  cnnlCreateTensorDescriptor(&coarse_centroids_desc_);
  { const int dim[2] = { nlist, d_ }; SetDesc(coarse_centroids_desc_, 2, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT);}
  cnnlCreateTensorDescriptor(&pq_centroids_desc_);
  cnnlCreateTensorDescriptor(&nlist_size_desc_);
  cnnlCreateTensorDescriptor(&codes_ids_ptr_desc_);
  const int dim[1] = { nlist };
  SetDesc(nlist_size_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);
  SetDesc(codes_ids_ptr_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);

  cnnlCreateTensorDescriptor(&query_vectors_desc_);
  cnnlCreateTensorDescriptor(&query_residuals_desc_);
  cnnlCreateTensorDescriptor(&nprobe_indices_desc_);
  cnnlCreateTensorDescriptor(&topk_distances_desc_);
  cnnlCreateTensorDescriptor(&topk_ids_desc_);

  cnnlCreateTensorDescriptor(&add_vectors_desc_);
  cnnlCreateTensorDescriptor(&add_ids_desc_);
  cnnlCreateTensorDescriptor(&inserts_idx_desc_);
  cnnlCreateTensorDescriptor(&inserts_size_desc_);

  cnnlCreateTensorDescriptor(&remove_codes_desc_);
  cnnlCreateTensorDescriptor(&remove_ids_desc_);

  // cnrtSetDeviceFlag(CNRT_QUEUE_SYNC_YIELD);
  cnrtQueueCreate(&cnrt_queue_);
  cnnlCreate(&cnnl_handle_);
  cnnlSetQueue(cnnl_handle_, cnrt_queue_);

  if (pipeline_mode) {
    thread_ = std::thread([this] { DeviceGuard(device_id_); while (TaskExecutor() >= 0); });
  }
}

IVFPQ3::~IVFPQ3() {
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
  nlist_alloc_size_.clear();
  vectors_ptr_.clear();
  ids_ptr_.clear();

  if (coarse_centroids_desc_) cnnlDestroyTensorDescriptor(coarse_centroids_desc_);
  if (pq_centroids_desc_) cnnlDestroyTensorDescriptor(pq_centroids_desc_);
  if (nlist_size_desc_) cnnlDestroyTensorDescriptor(nlist_size_desc_);
  if (codes_ids_ptr_desc_) cnnlDestroyTensorDescriptor(codes_ids_ptr_desc_);

  if (query_vectors_desc_) cnnlDestroyTensorDescriptor(query_vectors_desc_);
  if (query_residuals_desc_) cnnlDestroyTensorDescriptor(query_residuals_desc_);
  if (nprobe_indices_desc_) cnnlDestroyTensorDescriptor(nprobe_indices_desc_);
  if (topk_distances_desc_) cnnlDestroyTensorDescriptor(topk_distances_desc_);
  if (topk_ids_desc_) cnnlDestroyTensorDescriptor(topk_ids_desc_);

  if (add_vectors_desc_) cnnlDestroyTensorDescriptor(add_vectors_desc_);
  if (add_ids_desc_) cnnlDestroyTensorDescriptor(add_ids_desc_);
  if (inserts_idx_desc_) cnnlDestroyTensorDescriptor(inserts_idx_desc_);
  if (inserts_size_desc_) cnnlDestroyTensorDescriptor(inserts_size_desc_);

  if (remove_codes_desc_) cnnlDestroyTensorDescriptor(remove_codes_desc_);
  if (remove_ids_desc_) cnnlDestroyTensorDescriptor(remove_ids_desc_);

  if (cnnl_handle_) cnnlDestroy(cnnl_handle_);
  if (cnrt_queue_) cnrtQueueDestroy(cnrt_queue_);

  LOGC(IVFPQ) << "\033[35mIVFPQ3::~IVFPQ3()\033[0m";
}

cnindexReturn_t IVFPQ3::Reset() {
  LOGT(IVFPQ) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  for (const auto &p : vectors_ptr_) FreeMLUMemory(p);

  int nlist = nlist_;
  nlist_size_.assign(nlist, 0);
  nlist_alloc_size_.assign(nlist, 0);
  vectors_ptr_.assign(nlist, nullptr);
  ids_ptr_.assign(nlist, nullptr);

  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, vectors_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  ntotal_ = 0;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ3::SetCentroids(const float *centroids) {
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
  { int d[3] = { ksub_, M_, dsub_ }; SetDesc(pq_centroids_desc_, 3, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  cnrtMemcpy(pq_centroids_mlu_, centroids_trans.data(), sizeof(float) * ksub_ * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ3::SetListData(int index, int size, const void *codes, const int *ids) {
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

  int list_size = size;
  size_t list_bytes = (size_t)list_size * vector_size_;
  std::vector<uint8_t> codes_trans;
  // calculate allocate size.
  int alloc_size = CeilPower2(list_size);
  alloc_size = std::max(list_size, std::min(alloc_size, op_limit_size_));
  nlist_alloc_size_[index] = alloc_size;
  size_t codes_size = ALIGN_128(sizeof(uint8_t) * (size_t)alloc_size * vector_size_);
  size_t ids_size = ALIGN_128(sizeof(int) * (size_t)alloc_size);
  // allocate for codes and ids.
  FreeMLUMemory(vectors_ptr_[index]);
  vectors_ptr_[index] = AllocMLUMemory(codes_size + ids_size);
  if (!vectors_ptr_[index]) return CNINDEX_RET_ALLOC_FAILED;
  codes_trans.resize(list_bytes);
  const uint8_t *list_codes = static_cast<const uint8_t *>(codes);
  // trans codes: [list_size, code_size] -> [code_size, list_size].
  for (size_t i = 0; i < (size_t)list_size; i++) {
    for (int j = 0; j < vector_size_; j++) {
      codes_trans[j * (size_t)list_size + i] = list_codes[i * vector_size_ + j];
    }
  }
  cnrtMemcpy(vectors_ptr_[index], codes_trans.data(), list_bytes, CNRT_MEM_TRANS_DIR_HOST2DEV);
  ids_ptr_[index] = static_cast<uint8_t *>(vectors_ptr_[index]) + codes_size;
  if (!ids_ptr_[index]) return CNINDEX_RET_ALLOC_FAILED;
  int *list_ids = const_cast<int *>(ids);
  cnrtMemcpy(ids_ptr_[index], list_ids, sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_HOST2DEV);

  ntotal_ = ntotal_ - nlist_size_[index] + size;
  nlist_size_[index] = size;

  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist_, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, vectors_ptr_.data(), sizeof(void *) * nlist_, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

int IVFPQ3::GetListSize(int index) const {
  LOGT(IVFPQ) << "GetListSize(" << index << ")";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  if (index >= nlist_ || index < 0) {
    LOGE(IVFPQ) << "GetListSize() invalid list index: " << index;
    return CNINDEX_RET_BAD_PARAMS;
  }

  return nlist_size_[index];
}

cnindexReturn_t IVFPQ3::GetListData(int index, void *codes, int *ids) const {
  LOGT(IVFPQ) << "GetListData(" << index << ", " << static_cast<void *>(codes) << ", "
              << static_cast<void *>(ids) << ")";

  if (index >= nlist_ || index < 0) {
    LOGE(IVFPQ) << "GetListData() invalid list index: " << index;
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  std::vector<uint8_t> codes_trans;
  int list_size = nlist_size_[index];
  if (list_size <= 0) {
    LOGD(IVFPQ) << "GetListData() list[" << index << "] is empty";
    return CNINDEX_RET_SUCCESS;
  }

  if (codes) {
    if (!vectors_ptr_[index]) return CNINDEX_RET_NOT_VALID;
    size_t list_bytes = (size_t)list_size * vector_size_;
    uint8_t *list_codes = static_cast<uint8_t *>(codes);
    codes_trans.resize(list_bytes);
    cnrtMemcpy(codes_trans.data(), vectors_ptr_[index], sizeof(uint8_t) * list_bytes, CNRT_MEM_TRANS_DIR_DEV2HOST);
    // trans codes: [code_size, list_size] -> [list_size, code_size].
    for (int i = 0; i < vector_size_; i++) {
      for (size_t j = 0; j < (size_t)list_size; j++) {
        list_codes[j * vector_size_ + i] = codes_trans[i * (size_t)list_size + j];
      }
    }
  }
  if (ids) {
    if (!ids_ptr_[index]) return CNINDEX_RET_NOT_VALID;
    cnrtMemcpy(ids, ids_ptr_[index], sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ3::Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const {
  LOGT(IVFPQ) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << nprobe << ", " << k << ", "
              << static_cast<void *>(ids) << ", " << static_cast<void *>(distances) << ")";

  if (n <= 0 || !x || nprobe <= 0 || !ids) {
    LOGE(IVFPQ) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if ( k <= 0 || k > 1200) {
    LOGE(IVFPQ) << "Search() invalid k=" << k;
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (!coarse_centroids_mlu_ || !pq_centroids_mlu_) {
    LOGE(IVFPQ) << "Search() centroids is invalid";
    return CNINDEX_RET_NOT_VALID;
  }
  if (ntotal_ == 0) {
    LOGE(IVFPQ) << "Search() no vector";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  nprobe = std::min(nprobe, nlist_);
  int residual_nprobe = nlist_ == 1 ? core_number_ : nprobe;
  bool pipeline_mode = flat_->flat_->IsCPUImpl() && pipeline_mode_;

  int ndeal = 2048;  // d=1024, m=64, ntotal=8M
  if (ntotal_ > (1 << 23)) ndeal /= (ntotal_ >> 23);
  if (ndeal == 0) ndeal = 1;
  ndeal = pipeline_mode ? 1 : (ndeal > n ? n : ndeal);

  // LOGE(IVFPQ) << "Search() ndeal=" << ndeal;

  { int d[2] = { ndeal, nprobe }; SetDesc(nprobe_indices_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
  { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
  size_t workspace_size;
  cnnlGetPqSearchWorkspaceSize(cnnl_handle_, nprobe_indices_desc_, topk_distances_desc_, topk_ids_desc_,
                               &workspace_size, nlist_ == 1 ? 0 : 2);

  size_t vectors_size = ALIGN_128(sizeof(float) * (size_t)n * d_);
  size_t residuals_size = ALIGN_128(sizeof(float) * residual_nprobe * d_);  // op workspace
  size_t indices_size = ALIGN_128(sizeof(int) * (size_t)n * nprobe);
  size_t topk_distances_size = ALIGN_128(sizeof(float) * (size_t)n * k);
  size_t topk_ids_size = ALIGN_128(sizeof(int) * (size_t)n * k);
  size_t op_memory_size = vectors_size + residuals_size + indices_size + topk_distances_size + topk_ids_size;
  op_memory_size += workspace_size;
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
  void *workspace_mlu = op_memory_mlu;
  void *vectors_mlu = static_cast<uint8_t *>(workspace_mlu) + workspace_size;
  void *residuals_mlu = static_cast<uint8_t *>(vectors_mlu) + vectors_size;
  void *indices_mlu = static_cast<uint8_t *>(residuals_mlu) + residuals_size;
  void *topk_distances_mlu = static_cast<uint8_t *>(indices_mlu) + indices_size;
  void *topk_ids_mlu = static_cast<uint8_t *>(topk_distances_mlu) + topk_distances_size;

  std::vector<int> indices;
  int ret = CNINDEX_RET_SUCCESS;

  if (pipeline_mode) {
    int64_t dealed = 0;
    std::promise<int> promise;
    indices.resize(ndeal * nprobe);

    while (dealed < n) {
      if (exit_) return CNINDEX_RET_NOT_VALID;
      if ((dealed + ndeal) > n) ndeal = n - dealed;
      const float *x_t = x + dealed * d_;

      // prepare indices.
      if (nlist_ == 1) {
        if (indices.empty()) indices.assign(ndeal * nprobe, 0);
      } else {
        if (indices.empty()) indices.resize(ndeal * nprobe);
        if (CNINDEX_RET_SUCCESS != flat_->Search(ndeal, x_t, nprobe, indices.data(), nullptr)) {
          LOGE(IVFPQ) << "Search() coarse search failed";
          break;
        }
      }

      // copy to MLU
      float *vectors_t = static_cast<float *>(vectors_mlu) + dealed * d_;
      int *indices_t = static_cast<int *>(indices_mlu) + dealed * nprobe;
      cnrtMemcpy(vectors_t, const_cast<float *>(x_t), sizeof(float) * ndeal * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
      cnrtMemcpy(indices_t, indices.data(), sizeof(int) * ndeal * nprobe, CNRT_MEM_TRANS_DIR_HOST2DEV);

      int *ids_t = static_cast<int *>(topk_ids_mlu) + dealed * k;
      float *distances_t = static_cast<float *>(topk_distances_mlu) + dealed * k;

      std::unique_lock<std::mutex> qlk(queue_mutex_);
      queue_full_.wait(qlk, [this] { return (exit_ || queue_.size() < queue_capacity_); } );
      if (exit_) return CNINDEX_RET_NOT_VALID;

      std::promise<int> *p_promise = (dealed + ndeal) >= n ? &promise : nullptr;
      Task task{ ndeal, vectors_t, residuals_mlu, nprobe, indices_t, k, workspace_mlu, workspace_size,
                 ids_t, distances_t, p_promise };
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
  } else {
    // prepare indices.
    if (nlist_ == 1) {
      indices.assign((size_t)n * nprobe, 0);
      cnrtMemcpy(indices_mlu, indices.data(), sizeof(int) * (size_t)n * nprobe, CNRT_MEM_TRANS_DIR_HOST2DEV);
    } else {
      if (flat_->flat_->IsCPUImpl()) {
        indices.resize((size_t)n * nprobe);
        cnindexReturn_t ret = flat_->Search(n, x, nprobe, indices.data(), nullptr);
        if (CNINDEX_RET_SUCCESS != ret) {
          LOGE(IVFPQ) << "Search() CPU coarse search failed";
          return ret;
        }
        cnrtMemcpy(indices_mlu, indices.data(), sizeof(int) * (size_t)n * nprobe, CNRT_MEM_TRANS_DIR_HOST2DEV);
      } else {
        cnindexReturn_t ret = flat_->flat_->Search(n, x, nprobe, reinterpret_cast<int *>(indices_mlu), nullptr, true);
        if (CNINDEX_RET_SUCCESS != ret) {
          LOGE(IVFPQ) << "Search() MLU coarse search failed";
          return ret;
        }
      }
    }

    cnrtMemcpy(vectors_mlu, const_cast<float *>(x), sizeof(float) * (size_t)n * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

    int64_t dealed = 0;
    while (dealed < n) {
      if (exit_) return CNINDEX_RET_NOT_VALID;
      if ((dealed + ndeal) > n) ndeal = n - dealed;

      { int d[2] = { ndeal, d_ }; SetDesc(query_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
      { int d[2] = { residual_nprobe, d_ }; SetDesc(query_residuals_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
      { int d[2] = { ndeal, nprobe }; SetDesc(nprobe_indices_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
      { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
      { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

      void *vectors_mlu_t = static_cast<float *>(vectors_mlu) + dealed * d_;
      void *indices_mlu_t = static_cast<int *>(indices_mlu) + dealed * nprobe;
      void *topk_distances_mlu_t = static_cast<float *>(topk_distances_mlu) + dealed * k;
      void *topk_ids_mlu_t = static_cast<int *>(topk_ids_mlu) + dealed * k;

      cnnlStatus_t status = cnnlPqSearch_v2(cnnl_handle_,
                                            query_vectors_desc_, vectors_mlu_t,
                                            query_residuals_desc_, residuals_mlu,
                                            coarse_centroids_desc_, coarse_centroids_mlu_,
                                            pq_centroids_desc_, pq_centroids_mlu_,
                                            nlist_size_desc_, nlist_size_mlu_,
                                            codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
                                            codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
                                            nprobe_indices_desc_, indices_mlu_t,
                                            workspace_mlu, workspace_size,
                                            topk_distances_desc_, topk_distances_mlu_t,
                                            topk_ids_desc_, topk_ids_mlu_t,
                                            nlist_ == 1 ? 0 : 2, ntotal_);
      // cnnlStatus_t status = cnnlPqSearch(cnnl_handle_,
      //                                    query_vectors_desc_, vectors_mlu_t,
      //                                    query_residuals_desc_, residuals_mlu,
      //                                    coarse_centroids_desc_, coarse_centroids_mlu_,
      //                                    pq_centroids_desc_, pq_centroids_mlu_,
      //                                    nlist_size_desc_, nlist_size_mlu_,
      //                                    codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
      //                                    codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
      //                                    nprobe_indices_desc_, indices_mlu_t,
      //                                    topk_distances_desc_, topk_distances_mlu_t,
      //                                    topk_ids_desc_, topk_ids_mlu_t, nlist_ == 1 ? 0 : 2);

      if (status != CNNL_STATUS_SUCCESS) {
        LOGE(IVFPQ) << "Search() invoke op failed";
        return CNINDEX_RET_OP_FAILED;
      }

      dealed += ndeal;
    }

    cnrtQueueSync(cnrt_queue_);

    cnrtMemcpy(ids, topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
    if (distances) cnrtMemcpy(distances, topk_distances_mlu, sizeof(float) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);

    return CNINDEX_RET_SUCCESS;
  }
}

int IVFPQ3::TaskExecutor() const {
  Task task;
  std::unique_lock<std::mutex> qlk(queue_mutex_);
  queue_empty_.wait(qlk, [this] { return (exit_ || !queue_.empty()); });
  if (exit_) {
    while (!queue_.empty()) {
      task = queue_.front();
      queue_.pop();
      if (task.promise) task.promise->set_value(CNINDEX_RET_OP_FAILED);
    }
    cnrtQueueSync(cnrt_queue_);
    return -1;
  }
  task = queue_.front();
  queue_.pop();
  qlk.unlock();
  queue_full_.notify_one();

  int residual_nprobe = nlist_ == 1 ? core_number_ : task.nprobe;

  { int d[2] = { task.n, d_ }; SetDesc(query_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { residual_nprobe, d_ }; SetDesc(query_residuals_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { task.n, task.nprobe }; SetDesc(nprobe_indices_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
  { int d[2] = { task.n, task.k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { task.n, task.k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

  cnnlStatus_t status = cnnlPqSearch_v2(cnnl_handle_,
                                        query_vectors_desc_, task.x,
                                        query_residuals_desc_, task.xr,
                                        coarse_centroids_desc_, coarse_centroids_mlu_,
                                        pq_centroids_desc_, pq_centroids_mlu_,
                                        nlist_size_desc_, nlist_size_mlu_,
                                        codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
                                        codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
                                        nprobe_indices_desc_, task.indices,
                                        task.workspace, task.workspace_size,
                                        topk_distances_desc_, task.indices,
                                        topk_ids_desc_, task.distances,
                                        nlist_ == 1 ? 0 : 2, ntotal_);

  // cnnlStatus_t status = cnnlPqSearch(cnnl_handle_,
  //                                    query_vectors_desc_, task.x,
  //                                    query_residuals_desc_, task.xr,
  //                                    coarse_centroids_desc_, coarse_centroids_mlu_,
  //                                    pq_centroids_desc_, pq_centroids_mlu_,
  //                                    nlist_size_desc_, nlist_size_mlu_,
  //                                    codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
  //                                    codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
  //                                    nprobe_indices_desc_, task.indices,
  //                                    topk_distances_desc_, task.distances,
  //                                    topk_ids_desc_, task.ids, nlist_ == 1 ? 0 : 2);

  if (status != CNNL_STATUS_SUCCESS) {
    LOGE(IVFPQ) << "TaskExecutor() invoke op failed";
  }
  if (task.promise) {
    cnrtQueueSync(cnrt_queue_);
    task.promise->set_value(status == CNNL_STATUS_SUCCESS ? CNINDEX_RET_SUCCESS : CNINDEX_RET_OP_FAILED);
    return 1;
  }
  return 0;
}

cnindexReturn_t IVFPQ3::Add(int n, const float *x, const int *ids) {
  LOGT(IVFPQ) << "Add(" << n << ", " << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !x || !ids) {
    LOGE(IVFPQ) << "Add() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (!coarse_centroids_mlu_ || !pq_centroids_mlu_) {
    LOGE(IVFPQ) << "Add() centroids is empty";
    return CNINDEX_RET_NOT_VALID;
  }
  if (((size_t)ntotal_ + n) > std::numeric_limits<int>::max()) {
    LOGE(IVFPQ) << "Add() vectors number to be added over int_max(" << std::numeric_limits<int>::max() << ")";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  using IntPair = std::pair<int, int>;
  int nlist = nlist_size_.size();
  int ndeal = 1024;
  int64_t dealed = 0;

  size_t vectors_size = ALIGN_128(sizeof(float) * ndeal * d_);
  size_t residuals_size = ALIGN_128(sizeof(float) * ndeal * d_);
  size_t ids_size = ALIGN_128(sizeof(int) * ndeal);
  size_t codes_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t ids_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t list_idx_size = ALIGN_128(sizeof(int) * nlist);
  size_t insert_count_size = ALIGN_128(sizeof(int) * ndeal);
  size_t op_memory_size =
      vectors_size + residuals_size + ids_size + codes_ptr_size + ids_ptr_size + list_idx_size + insert_count_size;
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
  void *vectors_mlu = op_memory_mlu;
  void *residuals_mlu = static_cast<uint8_t *>(vectors_mlu) + vectors_size;
  void *ids_mlu = static_cast<uint8_t *>(residuals_mlu) + residuals_size;
  void *codes_ptr_mlu = static_cast<uint8_t *>(ids_mlu) + ids_size;
  void *ids_ptr_mlu = static_cast<uint8_t *>(codes_ptr_mlu) + codes_ptr_size;
  void *list_idx_mlu = static_cast<uint8_t *>(ids_ptr_mlu) + ids_ptr_size;
  void *insert_size_mlu = static_cast<uint8_t *>(list_idx_mlu) + list_idx_size;

  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;
    const float *x_d = x + dealed * d_;
    const int *ids_d = ids + dealed;

    std::vector<IntPair> inserts;  // [list_idx, insert_count]
    std::vector<float> x_vec;
    std::vector<int> ids_vec;

    if (nlist_ == 1) {
      inserts.emplace_back(0, ndeal);
    } else {
      std::vector<int> indices(ndeal);
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
      ids_d = ids_vec.data();
    }

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
    cnrtMemcpy(vectors_mlu, const_cast<float *>(x_d), sizeof(float) * ndeal * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_mlu, const_cast<int *>(ids_d), sizeof(int) * ndeal, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(codes_ptr_mlu, codes_ptr.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_ptr_mlu, ids_ptr.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(list_idx_mlu, inserts_idx.data(), sizeof(int) * lists_idx, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(insert_size_mlu, inserts_size.data(), sizeof(int) * inserts_count, CNRT_MEM_TRANS_DIR_HOST2DEV);

    { int d[2] = { ndeal, d_ }; SetDesc(add_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[1] = { ndeal }; SetDesc(add_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { lists_idx }; SetDesc(inserts_idx_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { inserts_count }; SetDesc(inserts_size_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlPqAdd(cnnl_handle_,
                                    add_vectors_desc_, vectors_mlu,
                                    add_vectors_desc_, residuals_mlu,
                                    add_ids_desc_, ids_mlu,
                                    coarse_centroids_desc_, coarse_centroids_mlu_,
                                    pq_centroids_desc_, pq_centroids_mlu_,
                                    codes_ids_ptr_desc_, (void **)codes_ptr_mlu_,
                                    codes_ids_ptr_desc_, (void **)ids_ptr_mlu_,
                                    codes_ids_ptr_desc_, (void **)codes_ptr_mlu,
                                    codes_ids_ptr_desc_, (void **)ids_ptr_mlu,
                                    inserts_idx_desc_, list_idx_mlu,
                                    inserts_size_desc_, insert_size_mlu,
                                    nlist_size_desc_, nlist_size_mlu_, nlist_ == 1 ? 0 : 2);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(IVFPQ) << "Add() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtQueueSync(cnrt_queue_);
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

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVFPQ3::Remove(int n, const int *ids) {
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

  std::vector<int> batch_size;
  std::vector<int> batch_offset;
#endif

  DeviceGuard(device_id_);

  int nlist = nlist_size_.size();
  for (int i = 0; i < n; i++) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    int id = ids[i];
    int list_idx = -1, offset = -1;

#ifdef USE_THREAD_POOL
    if (nlist == 1) {
      auto find_offset = [&batch_size, &batch_offset, this](int index, int id) -> int {
        int size = batch_size[index];
        std::vector<int> code_ids(size);
        DeviceGuard(device_id_);
        cnrtMemcpy(code_ids.data(), static_cast<int *>(ids_ptr_[0]) + batch_offset[index], sizeof(int) * (size_t)size,
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        auto it = std::find(code_ids.begin(), code_ids.end(), id);
        return (it != code_ids.end()) ? std::distance(code_ids.begin(), it) : -1;
      };

      int bs = ntotal_ / parallelism;
      if (batch_size.empty()) {
        batch_size.resize(parallelism);
        batch_offset.resize(parallelism);
        for (int i = 0; i < parallelism; i++) {
          batch_size[i] = i < (ntotal_ % parallelism) ? (bs + 1) : bs;
          batch_offset[i] = i == 0 ? 0 : batch_offset[i - 1] + batch_size[i - 1];
        }
      }

      std::vector<std::future<int>> fs;
      for (int j = 0; j < parallelism; j++) {
        fs.emplace_back(thread_pool->Push(find_offset, j, id));
      }
      for (auto &f : fs) {
        offset = f.get();
        if (offset != -1) {
          list_idx = 0;
          offset += batch_offset[std::distance(fs.data(), &f)];
          break;
        }
      }
    } else {
      auto find_offset = [this](int index, int id) -> int {
        int list_size = nlist_size_[index];
        std::vector<int> code_ids(list_size);
        DeviceGuard(device_id_);
        cnrtMemcpy(code_ids.data(), ids_ptr_[index], sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
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

    cnnlStatus_t status = cnnlPqRemove(cnnl_handle_,
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
      cnrtQueueSync(cnrt_queue_);
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

#endif  // ENABLE_MLU300

}  // impl

}  // cnindex
