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
#include "cnindex_pq.h"
#include "cnindex_pq_mlu300.h"

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

PQ3::PQ3(int d, cnindexMetric_t metric, int M, int nbits, int device_id)
    : PQ(d, metric, M, nbits, device_id), nallocated_(0), index_as_id_(0) {
  LOGC(PQ) << "\033[35mPQ3::PQ3()\033[0m";

  if (!(d_ == 256 || d_ == 512 || d_ == 768 || d_ == 1024) || !(M_ == 32 || M_ == 64) || nbits_ != 8) {
    LOGE(PQ) << "PQ3() bad parameters: d=" << d_ << ", M=" << M_ << ", nbits=" << nbits_;
    return;
  }

  int cluster_num, core_num_per_cluster;
  cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, device_id_);
  cnrtDeviceGetAttribute(&core_num_per_cluster, cnrtAttrMcorePerCluster, device_id_);
  core_number_ = cluster_num * core_num_per_cluster;
  int nram_size_per_core;
  cnrtDeviceGetAttribute(&nram_size_per_core, cnrtAttrNramSizePerMcore, device_id_);
  int reserved_nram_size = 128 << 10;
  int nram_size = nram_size_per_core - reserved_nram_size - 4 * 8 - 1;
  int m_align = (M_ / 64 + (int)(M_ % 64 > 0)) * 64;
  op_limit_size_ = (nram_size_per_core / m_align / code_size_) & ~0xff;

  DeviceGuard(device_id_);

  codes_ptr_ = nullptr;
  ids_ptr_ = nullptr;

  // alloc fixed memory
  size_t centroids_size = ALIGN_128(sizeof(float) * ksub_ * d_);
  size_t nlist_size = ALIGN_128(sizeof(int));
  size_t ptrs_size = ALIGN_128(sizeof(void *));
  size_t memory_size = centroids_size + nlist_size + 2 * ptrs_size;
 
  centroids_mlu_ = AllocMLUMemory(memory_size);
  if (!centroids_mlu_) return;
  cnrtMemset(centroids_mlu_, 0, memory_size);
  nlist_size_mlu_ = static_cast<uint8_t *>(centroids_mlu_) + centroids_size;
  codes_ptr_mlu_ = static_cast<uint8_t *>(nlist_size_mlu_) + nlist_size;
  ids_ptr_mlu_ = static_cast<uint8_t *>(codes_ptr_mlu_) + ptrs_size;

  // create descriptors
  cnnlCreateTensorDescriptor(&centroids_desc_);
  cnnlCreateTensorDescriptor(&nlist_size_desc_);
  cnnlCreateTensorDescriptor(&codes_ids_ptr_desc_);
  const int dim[1] = { 1 };
  SetDesc(nlist_size_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);
  SetDesc(codes_ids_ptr_desc_, 1, dim, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);

  cnnlCreateTensorDescriptor(&query_vectors_desc_);
  cnnlCreateTensorDescriptor(&query_residuals_desc_);
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
}

PQ3::~PQ3() {
  if (exit_) return;
  exit_ = true;

  std::lock_guard<std::mutex> lk(mutex_);

  DeviceGuard(device_id_);
  FreeMLUMemory(codes_ptr_);
  FreeMLUMemory(centroids_mlu_);
#ifndef USE_BFC_ALLOCATOR
  FreeMLUMemory(op_memory_mlu_);
#endif

  nallocated_ = 0;
  codes_ptr_ = nullptr;
  ids_ptr_ = nullptr;

  if (centroids_desc_) cnnlDestroyTensorDescriptor(centroids_desc_);
  if (nlist_size_desc_) cnnlDestroyTensorDescriptor(nlist_size_desc_);
  if (codes_ids_ptr_desc_) cnnlDestroyTensorDescriptor(codes_ids_ptr_desc_);

  if (query_vectors_desc_) cnnlDestroyTensorDescriptor(query_vectors_desc_);
  if (query_residuals_desc_) cnnlDestroyTensorDescriptor(query_residuals_desc_);
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

  LOGC(PQ) << "\033[35mIVFPQ3::~PQ3()\033[0m";
}

cnindexReturn_t PQ3::Reset() {
  LOGT(PQ) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);
  FreeMLUMemory(codes_ptr_);

  ntotal_ = 0;
  nallocated_ = 0;
  index_as_id_ = 0;
  codes_ptr_ = nullptr;
  ids_ptr_ = nullptr;

  cnrtMemcpy(nlist_size_mlu_, &ntotal_, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, &codes_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, &ids_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ3::SetCentroids(const float *centroids) {
  LOGT(PQ) << "SetCentroids(" << static_cast<const void *>(centroids) << ")";

  if (!centroids) {
    LOGE(PQ) << "SetCentroids() invalid parameters";
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

cnindexReturn_t PQ3::SetData(int size, const uint8_t *codes, const int *ids) {
  LOGT(PQ) << "SetData(" << size << ", " << static_cast<const void *>(codes) << ", "
              << static_cast<const void *>(ids) << ")";

  if (!codes || !ids) {
    LOGE(PQ) << "SetData() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (size <= 0) {
    LOGE(PQ) << "SetData() input is empty";
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  size_t codes_bytes = (size_t)size * code_size_;
  std::vector<uint8_t> codes_trans;
  // calculate allocate size.
  int alloc_size = CeilPower2(size);
  alloc_size = std::max(size, std::min(alloc_size, op_limit_size_));
  nallocated_ = alloc_size;
  size_t codes_size = ALIGN_128(sizeof(uint8_t) * alloc_size * code_size_);
  size_t ids_size = ALIGN_128(sizeof(int) * alloc_size);
  // allocate for codes and ids.
  FreeMLUMemory(codes_ptr_);
  codes_ptr_ = AllocMLUMemory(codes_size + ids_size);
  if (!codes_ptr_) return CNINDEX_RET_ALLOC_FAILED;
  codes_trans.resize(codes_bytes);
  // trans codes: [total_size, code_size] -> [code_size, total_size].
  for (size_t i = 0; i < (size_t)size; i++) {
    for (int j = 0; j < code_size_; j++) {
      codes_trans[j * (size_t)size + i] = codes[i * code_size_ + j];
    }
  }
  cnrtMemcpy(codes_ptr_, codes_trans.data(), codes_bytes, CNRT_MEM_TRANS_DIR_HOST2DEV);
  ids_ptr_ = static_cast<uint8_t *>(codes_ptr_) + codes_size;
  if (!ids_ptr_) return CNINDEX_RET_ALLOC_FAILED;
  cnrtMemcpy(ids_ptr_, const_cast<int *>(ids), sizeof(int) * size, CNRT_MEM_TRANS_DIR_HOST2DEV);

  ntotal_ = size;
  index_as_id_ = -1;

  cnrtMemcpy(nlist_size_mlu_, &ntotal_, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, &codes_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, &ids_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

int PQ3::GetSize() const {
  LOGT(PQ) << "GetSize()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  return ntotal_;
}

cnindexReturn_t PQ3::GetData(uint8_t *codes, int *ids) const {
  LOGT(PQ) << "GetData(" << static_cast<void *>(codes) << ", " << static_cast<void *>(ids) << ")";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  if (ntotal_ <= 0) {
    LOGI(PQ) << "GetData() no vectors";
    return CNINDEX_RET_NOT_VALID;
  }

  DeviceGuard(device_id_);

  std::vector<uint8_t> codes_trans;
  if (codes) {
    if (!codes_ptr_) return CNINDEX_RET_NOT_VALID;
    size_t codes_bytes = ntotal_ * code_size_;
    uint8_t *all_codes = static_cast<uint8_t *>(codes);
    codes_trans.resize(codes_bytes);
    cnrtMemcpy(codes_trans.data(), codes_ptr_, sizeof(uint8_t) * codes_bytes, CNRT_MEM_TRANS_DIR_DEV2HOST);
    // trans codes: [code_size, size] -> [size, code_size].
    for (int i = 0; i < code_size_; i++) {
      for (size_t j = 0; j < (size_t)ntotal_; j++) {
        all_codes[j * code_size_ + i] = codes_trans[i * (size_t)ntotal_ + j];
      }
    }
  }
  if (ids) {
    if (!ids_ptr_) return CNINDEX_RET_NOT_VALID;
    cnrtMemcpy(ids, ids_ptr_, sizeof(int) * (size_t)ntotal_, CNRT_MEM_TRANS_DIR_DEV2HOST);
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ3::Search(int n, const float *x, int k, int *ids, float *distances) const {
  LOGT(PQ) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << k << ", " << static_cast<void *>(ids)
           << ", " << static_cast<void *>(distances) << ")";

  if (n <= 0 || !x || !ids) {
    LOGE(PQ) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if ( k <= 0 || k > 1200) {
    LOGE(PQ) << "Search() invalid k=" << k;
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (!centroids_mlu_) {
    LOGE(PQ) << "Search() centroids is invalid";
    return CNINDEX_RET_NOT_VALID;
  }
  if (ntotal_ == 0) {
    LOGE(PQ) << "Search() no vector";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int ndeal = 2048;  // d=1024, m=64, ntotal=8M
  if (ntotal_ > (1 << 23)) ndeal /= (ntotal_ >> 23);
  if (ndeal == 0) ndeal = 1;
  ndeal = ndeal > n ? n : ndeal;

  // LOGE(IVFPQ) << "Search() ndeal=" << ndeal;

  { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
  size_t workspace_size;
  cnnlGetPqSearchWorkspaceSize(cnnl_handle_, nullptr, topk_distances_desc_, topk_ids_desc_, &workspace_size, 1);

  size_t vectors_size = ALIGN_128(sizeof(float) * (size_t)n * d_);
  size_t residuals_size = ALIGN_128(sizeof(float) * core_number_ * d_);
  size_t topk_distances_size = ALIGN_128(sizeof(float) * (size_t)n * k);
  size_t topk_ids_size = ALIGN_128(sizeof(int) * (size_t)n * k);
  size_t op_memory_size = workspace_size + vectors_size + residuals_size + topk_distances_size + topk_ids_size;
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
  void *topk_distances_mlu = static_cast<uint8_t *>(residuals_mlu) + residuals_size;
  void *topk_ids_mlu = static_cast<uint8_t *>(topk_distances_mlu) + topk_distances_size;

  cnrtMemcpy(vectors_mlu, const_cast<float *>(x), sizeof(float) * (size_t)n * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  int64_t dealed = 0;
  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;

    { int d[2] = { ndeal, d_ }; SetDesc(query_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { core_number_, d_ }; SetDesc(query_residuals_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    void *vectors_mlu_t = static_cast<float *>(vectors_mlu) + dealed * d_;
    void *topk_distances_mlu_t = static_cast<float *>(topk_distances_mlu) + dealed * k;
    void *topk_ids_mlu_t = static_cast<int *>(topk_ids_mlu) + dealed * k;

    cnnlStatus_t status = cnnlPqSearch_v2(cnnl_handle_,
                                          query_vectors_desc_, vectors_mlu_t,
                                          query_residuals_desc_, residuals_mlu,
                                          nullptr, nullptr,
                                          centroids_desc_, centroids_mlu_,
                                          nlist_size_desc_, nlist_size_mlu_,
                                          codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
                                          codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
                                          nullptr, nullptr,
                                          workspace_mlu, workspace_size,
                                          topk_distances_desc_, topk_distances_mlu_t,
                                          topk_ids_desc_, topk_ids_mlu_t,
                                          1, ntotal_);

    // cnnlStatus_t status = cnnlPqSearch(cnnl_handle_,
    //                                    query_vectors_desc_, vectors_mlu_t,
    //                                    query_residuals_desc_, residuals_mlu,
    //                                    nullptr, nullptr,
    //                                    centroids_desc_, centroids_mlu_,
    //                                    nlist_size_desc_, nlist_size_mlu_,
    //                                    codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
    //                                    codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
    //                                    nullptr, nullptr,
    //                                    topk_distances_desc_, topk_distances_mlu_t,
    //                                    topk_ids_desc_, topk_ids_mlu_t, 1);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(PQ) << "Search() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    }

    dealed += ndeal;
  }

  cnrtQueueSync(cnrt_queue_);
  cnrtMemcpy(ids, topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
  if (distances) cnrtMemcpy(distances, topk_distances_mlu, sizeof(float) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ3::Add(int n, const float *x, const int *ids) {
  LOGT(PQ) << "Add(" << n << ", " << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !x || !ids) {
    LOGE(PQ) << "Add() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (!centroids_mlu_) {
    LOGE(PQ) << "Add() centroids is empty";
    return CNINDEX_RET_NOT_VALID;
  }
  if (((size_t)ntotal_ + n) > std::numeric_limits<int>::max()) {
    LOGE(PQ) << "Add() vectors number to be added over int_max(" << std::numeric_limits<int>::max() << ")";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  std::vector<int> ids_gen;
  if (index_as_id_ == 0) index_as_id_ = !ids ? 1 : -1;
  if (index_as_id_ == -1 && !ids) {
    LOGE(PQ) << "Add() need ids";
    return CNINDEX_RET_BAD_PARAMS;
  } else if (index_as_id_ == 1) {
    if (ids) LOGW(PQ) << "Add() index as id, discard input ids";
    for (int i = 0; i < n; i++) ids_gen.push_back(ntotal_ + i);
    ids = ids_gen.data();
  }

  DeviceGuard(device_id_);

  int ndeal = 1024;
  int64_t dealed = 0;

  size_t vectors_size = ALIGN_128(sizeof(float) * ndeal * d_);
  size_t ids_size = ALIGN_128(sizeof(int) * ndeal);
  size_t codes_ptr_size = ALIGN_128(sizeof(void *));
  size_t ids_ptr_size = ALIGN_128(sizeof(void *));
  size_t list_idx_size = ALIGN_128(sizeof(int));
  size_t insert_count_size = ALIGN_128(sizeof(int) * ndeal);
  size_t op_memory_size = vectors_size + ids_size + codes_ptr_size + ids_ptr_size + list_idx_size + insert_count_size;
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
  void *ids_mlu = static_cast<uint8_t *>(vectors_mlu) + vectors_size;
  void *codes_ptr_mlu = static_cast<uint8_t *>(ids_mlu) + ids_size;
  void *ids_ptr_mlu = static_cast<uint8_t *>(codes_ptr_mlu) + codes_ptr_size;
  void *list_idx_mlu = static_cast<uint8_t *>(ids_ptr_mlu) + ids_ptr_size;
  void *insert_size_mlu = static_cast<uint8_t *>(list_idx_mlu) + list_idx_size;

  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;
    const float *x_d = x + dealed * d_;
    const int *ids_d = ids + dealed;

    // check if need reallocate memory
    void *codes_ptr = codes_ptr_;
    void *ids_ptr = ids_ptr_;
    int inserts_idx = 0, inserts_size = ndeal;
    int new_size = ntotal_ + ndeal;
    if (new_size > std::min(nallocated_, op_limit_size_)) {
      // calculate allocate size.
      int alloc_size = CeilPower2(new_size);
      alloc_size = std::max(new_size, std::min(alloc_size, op_limit_size_));
      size_t codes_size = ALIGN_128(code_size_ * (size_t)alloc_size);
      size_t ids_size = ALIGN_128(sizeof(int) * (size_t)alloc_size);
      // allocate for codes and ids.
      codes_ptr = AllocMLUMemory(codes_size + ids_size);
      if (!codes_ptr) return CNINDEX_RET_ALLOC_FAILED;
      ids_ptr = static_cast<uint8_t *>(codes_ptr) + codes_size;
      nallocated_ = alloc_size;
    }

    // copy to MLU
    cnrtMemcpy(vectors_mlu, const_cast<float *>(x_d), sizeof(float) * ndeal * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_mlu, const_cast<int *>(ids_d), sizeof(int) * ndeal, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(codes_ptr_mlu, &codes_ptr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(ids_ptr_mlu, &ids_ptr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(list_idx_mlu, &inserts_idx, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(insert_size_mlu, &inserts_size, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);

    { int d[2] = { ndeal, d_ }; SetDesc(add_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[1] = { ndeal }; SetDesc(add_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { 1 }; SetDesc(inserts_idx_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[1] = { 1 }; SetDesc(inserts_size_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlPqAdd(cnnl_handle_,
                                    add_vectors_desc_, vectors_mlu,
                                    nullptr, nullptr,
                                    add_ids_desc_, ids_mlu,
                                    nullptr, nullptr,
                                    centroids_desc_, centroids_mlu_,
                                    codes_ids_ptr_desc_, (void **)codes_ptr_mlu_,
                                    codes_ids_ptr_desc_, (void **)ids_ptr_mlu_,
                                    codes_ids_ptr_desc_, (void **)codes_ptr_mlu,
                                    codes_ids_ptr_desc_, (void **)ids_ptr_mlu,
                                    inserts_idx_desc_, list_idx_mlu,
                                    inserts_size_desc_, insert_size_mlu,
                                    nlist_size_desc_, nlist_size_mlu_, 1);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(PQ) << "Add() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtQueueSync(cnrt_queue_);
      if (codes_ptr != codes_ptr_) {
        FreeMLUMemory(codes_ptr_);
        codes_ptr_ = codes_ptr;
        ids_ptr_ = ids_ptr;
        cnrtMemcpy(codes_ptr_mlu_, &codes_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
        cnrtMemcpy(ids_ptr_mlu_, &ids_ptr_, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
      }
      cnrtMemcpy(&ntotal_, nlist_size_mlu_, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST);
      LOGD(PQ) << "Add() add " << ndeal << " vectors ok";
    }

    dealed += ndeal;
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ3::Remove(int n, const int *ids) {
  LOGT(PQ) << "Remove(" << n << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !ids) {
    LOGE(PQ) << "Remove() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (ntotal_ == 0) {
    LOGW(PQ) << "Remove() no vectors";
    return CNINDEX_RET_SUCCESS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

#ifdef USE_THREAD_POOL
  EqualityThreadPool *thread_pool = cnindex::GetThreadPool();
  size_t thread_pool_size = thread_pool->Size();
  int parallelism = 4;
  size_t threads_number = thread_pool_size + parallelism;
  threads_number = std::min(threads_number, GetCPUCoreNumber());
  if (threads_number > thread_pool_size) thread_pool->Resize(threads_number);

  std::vector<int> batch_size;
  std::vector<int> batch_offset;
#endif

  DeviceGuard(device_id_);

  auto update_ids = [this] {
    if (index_as_id_ <= 0) return;
    std::vector<int> ids;
    for (int i = 0; i < ntotal_; i++) ids.push_back(i);
    cnrtMemcpy(ids_ptr_, ids.data(), sizeof(int) * (size_t)ntotal_, CNRT_MEM_TRANS_DIR_HOST2DEV);
  };

  std::vector<int> indices;
  for (int i = 0; i < n; i++) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    int id = ids[i];
    int offset = -1;

    if (index_as_id_ == 1) {
      if (indices.empty()) {
        indices.assign(ids, ids + n);
        std::sort(indices.begin(), indices.end(), [](const int &x, const int &y) { return x > y; });
      }
      offset = indices[i];
      if (offset >= ntotal_) {
        LOGW(PQ) << "Remove() index: " << offset << " is over ntotal: " << ntotal_;
        continue;
      }
    } else {
#ifdef USE_THREAD_POOL
      auto find_offset = [&batch_size, &batch_offset, this](int index, int id) -> int {
        int size = batch_size[index];
        std::vector<int> code_ids(size);
        DeviceGuard(device_id_);
        cnrtMemcpy(code_ids.data(), static_cast<int *>(ids_ptr_) + batch_offset[index], sizeof(int) * (size_t)size,
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
          offset += batch_offset[std::distance(fs.data(), &f)];
          break;
        }
      }
#else
      std::vector<int> code_ids(ntotal_);
      cnrtMemcpy(code_ids.data(), ids_ptr_, sizeof(int) * (size_t)ntotal_, CNRT_MEM_TRANS_DIR_DEV2HOST);
      auto it = std::find(code_ids.begin(), code_ids.end(), id);
      if (it != code_ids.end()) {
        offset = std::distance(code_ids.begin(), it);
        break;
      }
#endif
    }

    if (offset == -1) {
      LOGE(PQ) << "Remove() find id: " <<  ids[i] << " failed";
      update_ids();
#ifdef USE_THREAD_POOL
      thread_pool->Resize(thread_pool_size);
#endif
      return CNINDEX_RET_NOT_VALID;
    }

    LOGD(PQ) << "Remove() find id[" << i << "]: " << ids[i] << " at offset: " << offset;

    { int d[2] = { code_size_, ntotal_ }; SetDesc(remove_codes_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_UINT8); }
    { int d[1] = { ntotal_ }; SetDesc(remove_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlPqRemove(cnnl_handle_,
                                       remove_codes_desc_, (uint8_t *)codes_ptr_,
                                       remove_ids_desc_, (int *)ids_ptr_,
                                       nlist_size_desc_, (int *)nlist_size_mlu_,
                                       0, offset);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(PQ) << "Remove() invoke op failed";
  #ifdef USE_THREAD_POOL
      thread_pool->Resize(thread_pool_size);
  #endif
      update_ids();
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtQueueSync(cnrt_queue_);
      LOGD(PQ) << "Remove() remove id: " << ids[i] << " ok";
      ntotal_--;
    }
  }
#ifdef USE_THREAD_POOL
  thread_pool->Resize(thread_pool_size);
#endif
  update_ids();
  return CNINDEX_RET_SUCCESS;
}

#endif  // ENABLE_MLU300

}  // impl

}  // cnindex
