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
#include "cnindex_pq_mlu200.h"

using cnindex::AllocMLUMemory;
using cnindex::FreeMLUMemory;
using cnindex::DeviceGuard;
using cnindex::CeilPower2;
using cnindex::GetCPUCoreNumber;
using cnindex::GetThreadPool;
using cnindex::EqualityThreadPool;

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU200

PQ2::PQ2(int d, cnindexMetric_t metric, int M, int nbits, int device_id)
    : PQ(d, metric, M, nbits, device_id), index_as_id_(0) {
  LOGC(PQ) << "\033[35mPQ2::PQ2()\033[0m";

  if (!(d_ == 256 || d_ == 512 || d_ == 768 || d_ == 1024) || !(M_ == 32 || M_ == 64) || nbits_ != 8) {
    LOGE(PQ) << "PQ2() bad parameters: d=" << d_ << ", M=" << M_ << ", nbits=" << nbits_;
    return;
  }

  CNRTInit();

#if CNRT_MAJOR_VERSION < 5
  cnrtDeviceInfo_t dev_info;
  cnrtGetDeviceInfo(&dev_info, device_id_);
  core_number_ = dev_info.core_num;
  op_limit_size_ = ((484 << 10) / 4 / code_size_) & ~0xff;
#else
  int cluster_num, core_num_per_cluster;
  cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, device_id_);
  cnrtDeviceGetAttribute(&core_num_per_cluster, cnrtAttrMcorePerCluster, device_id_);
  core_number_ = cluster_num * core_num_per_cluster;
  int nram_size_per_core;
  cnrtDeviceGetAttribute(&nram_size_per_core, cnrtAttrNramSizePerMcore, device_id_);
  nram_size_per_core = 484 << 10;
  op_limit_size_ = (nram_size_per_core / 4 / code_size_) & ~0xff;
#endif

  DeviceGuard(device_id_);

  int nlist = core_number_;
  nlist_size_.resize(nlist, 0);
  nlist_bytes_.resize(nlist, 0);
  nlist_alloc_size_.resize(nlist, 0);
  codes_ptr_.resize(nlist, nullptr);
  ids_ptr_.resize(nlist, nullptr);

  // alloc fixed memory
  size_t centroids_size = ALIGN_128(sizeof(float) * ksub_ * d_);
  size_t nlist_size = ALIGN_128(sizeof(int) * nlist);
  size_t ptrs_size = ALIGN_128(sizeof(void *) * nlist);
  size_t memory_size = centroids_size + 3 * nlist_size + 2 * ptrs_size;
  centroids_mlu_ = AllocMLUMemory(memory_size);
  if (!centroids_mlu_) return;
  cnrtMemset(centroids_mlu_, 0, memory_size);
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
}

PQ2::~PQ2() {
  if (exit_) return;
  exit_ = true;

  std::lock_guard<std::mutex> lk(mutex_);

  DeviceGuard(device_id_);
  for (const auto &p : codes_ptr_) FreeMLUMemory(p);
  FreeMLUMemory(centroids_mlu_);
#ifndef USE_BFC_ALLOCATOR
  FreeMLUMemory(op_memory_mlu_);
#endif

  nlist_size_.clear();
  nlist_bytes_.clear();
  nlist_alloc_size_.clear();
  codes_ptr_.clear();
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

  LOGC(PQ) << "\033[35mPQ2::~PQ2()\033[0m";
}

cnindexReturn_t PQ2::Reset() {
  LOGT(PQ) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  for (const auto &p : codes_ptr_) FreeMLUMemory(p);

  int nlist = nlist_size_.size();
  nlist_size_.assign(nlist, 0);
  nlist_bytes_.assign(nlist, 0);
  nlist_alloc_size_.assign(nlist, 0);
  codes_ptr_.assign(nlist, nullptr);
  ids_ptr_.assign(nlist, nullptr);

  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, codes_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  ntotal_ = 0;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ2::SetCentroids(const float *centroids) {
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

cnindexReturn_t PQ2::SetData(int size, const uint8_t *codes, const int *ids) {
  LOGT(PQ) << "SetData(" << size << ", " << static_cast<const void *>(codes) << ", " << static_cast<const void *>(ids)
           << ")";

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

  ntotal_ = size;
  size = ntotal_ / nlist_size_.size();
  int list_r = ntotal_ % nlist_size_.size();
  int list_offset = 0;
  std::vector<uint8_t> codes_trans;
  for (int i = 0; i < nlist_size_.size(); i++) {
    int list_size = i < list_r ? (size + 1) : size;
    nlist_size_[i] = list_size;
    if (list_size <= 0) continue;
    nlist_bytes_[i] = list_size * code_size_;
    // calculate allocate size.
    int alloc_size = CeilPower2(list_size);
    alloc_size = std::max(list_size, std::min(alloc_size, op_limit_size_));
    nlist_alloc_size_[i] = alloc_size;
    size_t codes_size = ALIGN_128(code_size_ * (size_t)alloc_size);
    size_t ids_size = ALIGN_128(sizeof(int) * (size_t)alloc_size);
    // allocate for codes and ids.
    FreeMLUMemory(codes_ptr_[i]);
    codes_ptr_[i] = AllocMLUMemory(codes_size + ids_size);
    if (!codes_ptr_[i]) return CNINDEX_RET_ALLOC_FAILED;
    codes_trans.resize(nlist_bytes_[i]);
    const uint8_t *list_codes = codes + (size_t)list_offset * code_size_;
    // trans codes: [list_size, code_size] -> [code_size, list_size].
    for (size_t i = 0; i < (size_t)list_size; i++) {
      for (int j = 0; j < code_size_; j++) {
        codes_trans[j * (size_t)list_size + i] = list_codes[i * code_size_ + j];
      }
    }
    cnrtMemcpy(codes_ptr_[i], codes_trans.data(), nlist_bytes_[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
    ids_ptr_[i] = static_cast<uint8_t *>(codes_ptr_[i]) + codes_size;
    if (!ids_ptr_[i]) return CNINDEX_RET_ALLOC_FAILED;
    int *list_ids = const_cast<int *>(ids) + list_offset;
    cnrtMemcpy(ids_ptr_[i], list_ids, sizeof(int) * list_size, CNRT_MEM_TRANS_DIR_HOST2DEV);

    list_offset += list_size;
  }

  int nlist = nlist_size_.size();
  cnrtMemcpy(nlist_size_mlu_, nlist_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(codes_ptr_mlu_, codes_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

int PQ2::GetSize() const {
  LOGT(PQ) << "GetSize()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  return ntotal_;
}

cnindexReturn_t PQ2::GetData(uint8_t *codes, int *ids) const {
  LOGT(PQ) << "GetData(" << static_cast<void *>(codes) << ", " << static_cast<void *>(ids) << ")";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  if (ntotal_ <= 0) {
    LOGI(PQ) << "GetData() no vectors";
    return CNINDEX_RET_NOT_VALID;
  }

  DeviceGuard(device_id_);

  int offset = 0;
  std::vector<uint8_t> codes_trans;
  for (int i = 0; i < nlist_size_.size(); i++) {
    int list_size = nlist_size_[i];
    if (list_size <= 0) {
      continue;
    }
    if (codes) {
      if (!codes_ptr_[i]) return CNINDEX_RET_NOT_VALID;
      int list_bytes = nlist_bytes_[i];
      uint8_t *list_codes = codes + (size_t)offset * code_size_;
      codes_trans.resize(list_bytes);
      cnrtMemcpy(codes_trans.data(), codes_ptr_[i], sizeof(uint8_t) * list_bytes, CNRT_MEM_TRANS_DIR_DEV2HOST);
      // trans codes: [code_size, list_size] -> [list_size, code_size].
      for (int i = 0; i < code_size_; i++) {
        for (size_t j = 0; j < (size_t)list_size; j++) {
          list_codes[j * code_size_ + i] = codes_trans[i * (size_t)list_size + j];
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

cnindexReturn_t PQ2::Search(int n, const float *x, int k, int *ids, float *distances) const {
  LOGT(PQ) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << k << ", " << static_cast<void *>(ids)
           << ", " << static_cast<void *>(distances) << ")";

  if (n <= 0 || !x || k <= 0 || !ids) {
    LOGE(PQ) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }

  if (ntotal_ == 0) {
    LOGE(PQ) << "Search() no vector";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int nlist = nlist_size_.size();
  size_t vectors_size = ALIGN_128(sizeof(float) * (size_t)n * nlist * d_);
  size_t indices_size = ALIGN_128(sizeof(int) * (size_t)n * nlist);
  size_t topk_distances_size = ALIGN_128(sizeof(float) * (size_t)n * k);
  size_t topk_ids_size = ALIGN_128(sizeof(int) * (size_t)n * k);
  size_t op_memory_size = vectors_size + indices_size + topk_distances_size + topk_ids_size;
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
  void *indices_mlu = static_cast<uint8_t *>(vectors_mlu) + vectors_size;
  void *topk_distances_mlu = static_cast<uint8_t *>(indices_mlu) + indices_size;
  void *topk_ids_mlu = static_cast<uint8_t *>(topk_distances_mlu) + topk_distances_size;

  int ndeal = 1024, dealed = 0;
  std::vector<int> indices;
  std::vector<float> vectors(ndeal * nlist * d_);

  // prepare indices.
  for (int i = 0; i < nlist; i++) {
    indices.push_back(i);
  }
  for (int i = 1; i < ndeal; i++) {
    indices.insert(indices.end(), indices.begin(), indices.begin() + nlist);
  }

  // check if topk use brute force search.
  bool bfs = k > (ntotal_ / nlist);

  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;
    const float *x_d = x + dealed * d_;

    // prepare nlist vectors
    for (int i = 0; i < ndeal; i++) {
      for (int j = 0; j < nlist; j++) {
        memcpy(vectors.data() + (i * nlist + j) * d_, x_d + i * d_, sizeof(float) * d_);
      }
    }

    // copy to MLU
    float *vectors_d = static_cast<float *>(vectors_mlu) + dealed * nlist * d_;
    int *indices_d = static_cast<int *>(indices_mlu) + dealed * nlist;
    cnrtMemcpy(vectors_d, vectors.data(), sizeof(float) * ndeal * nlist * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(indices_d, indices.data(), sizeof(int) * ndeal * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

    int *ids_d = static_cast<int *>(topk_ids_mlu) + dealed * k;
    float *distances_d = static_cast<float *>(topk_distances_mlu) + dealed * k;

    { int d[3] = { ndeal, nlist, d_ }; SetDesc(nquery_desc_, 3, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, nlist }; SetDesc(nprobe_indices_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }
    { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlIvfProductQuantization(cnnl_handle_,
                                                     nquery_desc_, vectors_d,
                                                     centroids_desc_, centroids_mlu_,
                                                     nlist_size_desc_, nlist_size_mlu_,
                                                     nlist_size_desc_, nlist_bytes_mlu_,
                                                     nlist_size_desc_, nlist_alloc_size_mlu_,
                                                     codes_ids_ptr_desc_, (const void **)codes_ptr_mlu_,
                                                     codes_ids_ptr_desc_, (const void **)ids_ptr_mlu_,
                                                     nprobe_indices_desc_, indices_d,
                                                     topk_distances_desc_, distances_d,
                                                     topk_ids_desc_, ids_d, bfs);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(PQ) << "Search() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    }

    dealed += ndeal;
  }

  cnrtSyncQueue(cnrt_queue_);

  if (dealed >= n) {
    cnrtMemcpy(ids, topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
    if (distances) cnrtMemcpy(distances, topk_distances_mlu, sizeof(float) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ2::Add(int n, const float *x, const int *ids) {
  LOGT(PQ) << "Add(" << n << ", " << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !x) {
    LOGE(PQ) << "Add() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
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

  using IntPair = std::pair<int, int>;
  int nlist = nlist_size_.size();
  int ndeal = 1024;
  int64_t dealed = 0;

  size_t vectors_size = ALIGN_128(sizeof(float) * ndeal * d_);
  size_t ids_size = ALIGN_128(sizeof(int) * ndeal);
  size_t codes_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t ids_ptr_size = ALIGN_128(sizeof(void *) * nlist);
  size_t list_idx_size = ALIGN_128(sizeof(int) * nlist);
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

    std::vector<IntPair> inserts;  // [list_idx, insert_count]
    std::vector<int> indices;
    std::vector<float> x_vec;
    std::vector<int> ids_vec;

    // try to balance list size
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

    DeviceGuard(device_id_);

    // check if need reallocate memory
    std::vector<void *> codes_ptr(codes_ptr_);
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
        size_t codes_size = ALIGN_128(code_size_ * (size_t)alloc_size);
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
                                       add_residuals_desc_, vectors_mlu,
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
      LOGE(PQ) << "Add() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtSyncQueue(cnrt_queue_);
      for (const auto &idx : free_list_idx) {
        FreeMLUMemory(codes_ptr_[idx]);
        codes_ptr_[idx] = codes_ptr[idx];
        ids_ptr_[idx] = ids_ptr[idx];
      }
      if (!free_list_idx.empty()) {
        cnrtMemcpy(codes_ptr_mlu_, codes_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
        cnrtMemcpy(ids_ptr_mlu_, ids_ptr_.data(), sizeof(void *) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
      }
      ntotal_ += ndeal;
      cnrtMemcpy(nlist_size_.data(), nlist_size_mlu_, sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_DEV2HOST);
      LOGD(PQ) << "Add() add " << ndeal << " vectors ok";
    }

    dealed += ndeal;
  }

  for (int i = 0; i < nlist; i++) nlist_bytes_[i] = code_size_ * nlist_size_[i];
  cnrtMemcpy(nlist_bytes_mlu_, nlist_bytes_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(nlist_alloc_size_mlu_, nlist_alloc_size_.data(), sizeof(int) * nlist, CNRT_MEM_TRANS_DIR_HOST2DEV);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t PQ2::Remove(int n, const int *ids) {
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
  int parallelism = 4;
  EqualityThreadPool *thread_pool = cnindex::GetThreadPool();
  size_t thread_pool_size = thread_pool->Size();
  size_t threads_number = thread_pool_size + parallelism;
  threads_number = std::min(threads_number, GetCPUCoreNumber());
  if (threads_number > thread_pool_size) thread_pool->Resize(threads_number);
#endif

  DeviceGuard(device_id_);

  auto update_ids = [this] {
    if (index_as_id_ <= 0) return;
    std::vector<int> ids;
    for (int i = 0; i < ntotal_; i++) ids.push_back(i);
    int list_offset = 0;
    for (int i = 0; i < nlist_size_.size(); i++) {
      if (nlist_size_[i] <= 0) continue;
      int *list_ids = ids.data() + list_offset;
      cnrtMemcpy(ids_ptr_[i], list_ids, sizeof(int) * nlist_size_[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
      list_offset += nlist_size_[i];
    }
  };

  int nlist = nlist_size_.size();
  std::vector<int> indices;
  for (int i = 0; i < n; i++) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    int list_idx = -1, offset = -1;

    if (index_as_id_ == 1) {
      if (indices.empty()) {
        indices.assign(ids, ids + n);
        std::sort(indices.begin(), indices.end(), [](const int &x, const int &y) { return x > y; });
      }
      int index = indices[i];
      if (index >= ntotal_) {
        LOGW(PQ) << "Remove() index: " << index << " is over ntotal: " << ntotal_;
        continue;
      }
      for (auto &list_size : nlist_size_) {
        if (index >= list_size) {
          index -= list_size;
        } else {
          list_idx = std::distance(nlist_size_.data(), &list_size);
          offset = index;
          break;
        }
      }
    } else {
#ifdef USE_THREAD_POOL
      auto find_offset = [this](int index, int id) -> int {
        int list_size = nlist_size_[index];
        std::vector<int> code_ids(list_size);
        DeviceGuard(device_id_);
        cnrtMemcpy(code_ids.data(), ids_ptr_[index], sizeof(int) * (size_t)list_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
        auto it = std::find(code_ids.begin(), code_ids.end(), id);
        return (it != code_ids.end()) ? std::distance(code_ids.begin(), it) : -1;
      };

      int id = ids[i];
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
    }

    if (list_idx == -1 || offset == -1) {
      LOGE(PQ) << "Remove() find id: " <<  ids[i] << " failed";
      update_ids();
#ifdef USE_THREAD_POOL
      thread_pool->Resize(thread_pool_size);
#endif
      return CNINDEX_RET_NOT_VALID;
    }

    LOGD(PQ) << "Remove() find id[" << i << "]: " << ids[i] << " in list: " << list_idx << " offset: " << offset;

    int list_size = nlist_size_[list_idx];
    { int d[2] = { code_size_, list_size }; SetDesc(remove_codes_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_UINT8); }
    { int d[1] = { list_size }; SetDesc(remove_ids_desc_, 1, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    cnnlStatus_t status = cnnlIvfpqRemove(cnnl_handle_,
                                          remove_codes_desc_, (uint8_t *)codes_ptr_[list_idx],
                                          remove_ids_desc_, (int *)ids_ptr_[list_idx],
                                          nlist_size_desc_, (int *)nlist_size_mlu_,
                                          list_idx, offset);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(PQ) << "Remove() invoke op failed";
#ifdef USE_THREAD_POOL
      thread_pool->Resize(thread_pool_size);
#endif
      update_ids();
      return CNINDEX_RET_OP_FAILED;
    } else {
      cnrtSyncQueue(cnrt_queue_);
      LOGD(PQ) << "Remove() remove id: " << ids[i] << " ok";
      ntotal_--;
      nlist_size_[list_idx]--;
    }
  }
#ifdef USE_THREAD_POOL
  thread_pool->Resize(thread_pool_size);
#endif
  update_ids();
  return CNINDEX_RET_SUCCESS;
}

#endif  // ENABLE_MLU200

}  // impl

}  // cnindex
