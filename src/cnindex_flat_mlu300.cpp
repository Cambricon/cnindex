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
#include <memory>
#include <mutex>
#include <vector>

#include <cnrt.h>
#include <cnnl.h>
#include <cnnl_extra.h>

#include "utils/arithmetics.h"
#include "utils/log.h"
#include "utils/thread_pool.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_flat_mlu300.h"

using cnindex::AllocMLUMemory;
using cnindex::FreeMLUMemory;
using cnindex::DeviceGuard;

namespace cnindex {

namespace impl {

#ifdef ENABLE_MLU300
#define SRAM_MAX_INPUT_SIZE (256 * 1024)
#define SRAM_MAX_TOPK_CACHE (1024 * 1024)
#define MAX_NRAM_SIZE_USER (735 * 1024)
int Flat3::GetMaxBatch(int batch,
                       int dimension,
                       int k,
                       int loop_deal_nlib) const {

    //loop_deal_nlib 这个取值64
    //dimension为d，其值可取256、512、768、1024
    int pad_size = 32;
    int k_pad = (k + pad_size - 1) / pad_size  * pad_size;
    int max_deal_batch_input_sram = SRAM_MAX_INPUT_SIZE / dimension / sizeof(float);
    int max_deal_batch_Topk_sram = SRAM_MAX_TOPK_CACHE / k_pad / sizeof(float) / 2;
    int max_deal_batch = max_deal_batch_input_sram > max_deal_batch_Topk_sram ?
                         max_deal_batch_Topk_sram : max_deal_batch_input_sram;
    max_deal_batch = (max_deal_batch > batch) ? batch : max_deal_batch;
    int max_deal_batch_nram = MAX_NRAM_SIZE_USER / sizeof(float) / (dimension + loop_deal_nlib * 2);
    max_deal_batch = (max_deal_batch > max_deal_batch_nram) ? max_deal_batch_nram : max_deal_batch;
    max_deal_batch = max_deal_batch * cluster_num_ / 2;
    return max_deal_batch;
}

Flat3::Flat3(int d, cnindexMetric_t metric, int device_id)
    : Flat(d, metric), device_id_(device_id) {

  LOGC(Flat) << "\033[35mFlat3::Flat3(" << (metric_ == CNINDEX_METRIC_L2 ? "L2" : "IP") << ")\033[0m";

  if (d_ > 8000) {
    LOGE(Flat) << "Flat3() bad parameters: d=" << d_;
    return;
  }

  ntotal_ = 0;
  nallocated_ = 0;

  int cluster_num, core_num_per_cluster;
  cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, device_id_);
  cnrtDeviceGetAttribute(&core_num_per_cluster, cnrtAttrMcorePerCluster, device_id_);
  core_number_ = cluster_num * core_num_per_cluster;
  cluster_num_ = cluster_num;

  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (device_id_ >= dev_num) {
    LOGE(Flat) << "Flat3() invalid device id: " << device_id_;
    return;
  }

  DeviceGuard(device_id_);

  cnnlDistanceMode_t distance_mode = CNNL_DISTANCE_L2;
  if (metric_ == CNINDEX_METRIC_L2) {
    distance_mode = CNNL_DISTANCE_L2;
  } else if (metric_ == CNINDEX_METRIC_IP) {
    distance_mode = CNNL_DISTANCE_IP;
  } else {
    LOGE(Flat) << "Flat3() unsupported metric type: " << metric;
    return;
  }  

  cnnlCreateFlatSearchDescriptor(&search_desc_);
  cnnlCreateTensorDescriptor(&vectors_base_desc_);
  cnnlCreateTensorDescriptor(&query_vectors_desc_);
  cnnlCreateTensorDescriptor(&topk_distances_desc_);
  cnnlCreateTensorDescriptor(&topk_ids_desc_);

  cnnlSetFlatSearchDescriptor(search_desc_, distance_mode);

  // cnrtSetDeviceFlag(CNRT_QUEUE_SYNC_YIELD);
  cnrtQueueCreate(&cnrt_queue_);
  cnnlCreate(&cnnl_handle_);
  cnnlSetQueue(cnnl_handle_, cnrt_queue_);
}

Flat3::~Flat3() {
  if (exit_) return;
  exit_ = true;

  std::lock_guard<std::mutex> lk(mutex_);

  DeviceGuard(device_id_);
  FreeMLUMemory(vectors_base_mlu_);
#ifndef USE_BFC_ALLOCATOR
  FreeMLUMemory(op_memory_mlu_);
#endif

  if (search_desc_) cnnlDestroyFlatSearchDescriptor(search_desc_);
  if (vectors_base_desc_) cnnlDestroyTensorDescriptor(vectors_base_desc_);
  if (query_vectors_desc_) cnnlDestroyTensorDescriptor(query_vectors_desc_);
  if (topk_distances_desc_) cnnlDestroyTensorDescriptor(topk_distances_desc_);
  if (topk_ids_desc_) cnnlDestroyTensorDescriptor(topk_ids_desc_);

  if (cnnl_handle_) cnnlDestroy(cnnl_handle_);
  if (cnrt_queue_) cnrtQueueDestroy(cnrt_queue_);

  LOGC(Flat) << "\033[35mFlat3::~Flat3()\033[0m";
}

cnindexReturn_t Flat3::Reset() {
  LOGT(Flat) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  ids_.clear();
  if (vectors_base_mlu_) FreeMLUMemory(vectors_base_mlu_);
  vectors_base_mlu_ = nullptr;

  ntotal_ = 0;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t Flat3::Search(int n, const float *x, int k, int *ids, float *distances, bool output_on_mlu) const {
  LOGT(Flat) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << k << ", "
             << static_cast<void *>(ids) << ", " << static_cast<void *>(distances) << ", " << output_on_mlu << ")";

  if (n <= 0 || !x || !ids) {
    LOGE(Flat) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if ( k <= 0 || k > 1200) {
    LOGE(Flat) << "Search() invalid k=" << k;
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (ntotal_ == 0) {
    LOGE(Flat) << "Search() database is empty";
    return CNINDEX_RET_NOT_VALID;
  }
  if (output_on_mlu && !ids_.empty()) {
    LOGE(Flat) << "Search() only support output results on mlu with indices";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int ndeal = n;
  if (metric_ == CNINDEX_METRIC_L2) {
    ndeal = core_number_ * 32;  // d = 1024, ntotal=1M, topk=1000
    ndeal = (int64_t)ndeal * 1024 * (1 << 20) / d_ / ntotal_;  // d, ntotal as variable
    if (ndeal == 0) {
      ndeal = 1;
    } else if (ndeal > n) {
      ndeal = n;
    }
  } else {
    // ndeal = (ndeal > 5000 && ntotal_ > 1E7) ? 5000 : ndeal;
    ndeal = Flat3::GetMaxBatch(n, d_, k, 64);
  }

  // LOGE(Flat) << "Search() ndeal=" << ndeal;

  { int d[2] = { ntotal_, d_ }; SetDesc(vectors_base_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { ndeal, d_ }; SetDesc(query_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
  { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

  size_t query_vectors_size = ALIGN_128(sizeof(int) * (size_t)n * d_);
  size_t workspace_size;
  cnnlGetFlatSearchWorkspaceSize_v2(cnnl_handle_, search_desc_, vectors_base_desc_, topk_distances_desc_,
                                    topk_ids_desc_, &workspace_size);
  // cnnlGetFlatSearchWorkspaceSize(cnnl_handle_, vectors_base_desc_, topk_distances_desc_, topk_ids_desc_,
  //                                &workspace_size);
  workspace_size = ALIGN_128(workspace_size);
  size_t topk_distances_size = ALIGN_128(sizeof(float) * (size_t)n * k);
  size_t topk_ids_size = ALIGN_128(sizeof(int) * (size_t)n * k);
  size_t op_memory_size = query_vectors_size + workspace_size;
  if (!output_on_mlu || !distances) op_memory_size += topk_distances_size;
  if (!output_on_mlu) op_memory_size += topk_ids_size;
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
  void *query_vectors_mlu = op_memory_mlu;
  void *workspace_mlu = static_cast<uint8_t *>(query_vectors_mlu) + query_vectors_size;
  void *distances_mlu = static_cast<uint8_t *>(workspace_mlu) + workspace_size;
  void *topk_distances_mlu = (output_on_mlu && distances) ? reinterpret_cast<uint8_t *>(distances) : distances_mlu;
  void *ids_mlu = static_cast<uint8_t *>(distances_mlu) + topk_distances_size;
  void *topk_ids_mlu = output_on_mlu ? reinterpret_cast<uint8_t *>(ids) : ids_mlu;

  cnrtMemcpy(query_vectors_mlu, const_cast<float *>(x), sizeof(float) * n * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  int64_t dealed = 0;
  while (dealed < n) {
    if (exit_) return CNINDEX_RET_NOT_VALID;
    if ((dealed + ndeal) > n) ndeal = n - dealed;

    { int d[2] = { ndeal, d_ }; SetDesc(query_vectors_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, k }; SetDesc(topk_distances_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT); }
    { int d[2] = { ndeal, k }; SetDesc(topk_ids_desc_, 2, d, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32); }

    void *query_vectors_mlu_t = static_cast<float *>(query_vectors_mlu) + dealed * d_;
    void *topk_distances_mlu_t = static_cast<float *>(topk_distances_mlu) + dealed * k;
    void *topk_ids_mlu_t = static_cast<int *>(topk_ids_mlu) + dealed * k;

    cnnlStatus_t status = cnnlFlatSearch_v2(cnnl_handle_, search_desc_,
                                            vectors_base_desc_, vectors_base_mlu_,
                                            query_vectors_desc_, query_vectors_mlu_t,
                                            topk_distances_desc_, topk_distances_mlu_t,
                                            topk_ids_desc_, topk_ids_mlu_t,
                                            workspace_mlu, workspace_size);
    // cnnlStatus_t status = cnnlFlatSearch(cnnl_handle_,
    //                                      vectors_base_desc_, vectors_base_mlu_,
    //                                      query_vectors_desc_, query_vectors_mlu_t,
    //                                      workspace_mlu, workspace_size,
    //                                      topk_distances_desc_, topk_distances_mlu_t,
    //                                      topk_ids_desc_, topk_ids_mlu_t);

    if (status != CNNL_STATUS_SUCCESS) {
      LOGE(Flat) << "Search() invoke op failed";
      return CNINDEX_RET_OP_FAILED;
    }

    dealed += ndeal;
  }

  cnrtQueueSync(cnrt_queue_);

  if (!output_on_mlu) {
    if (!ids_.empty()) {
      std::vector<int> indices((size_t)n * k);
      cnrtMemcpy(indices.data(), topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
      for (int i = 0; i < indices.size(); i++) ids[i] = indices[i] < 0 ? -1 : ids_[indices[i]];
    } else {
      cnrtMemcpy(ids, topk_ids_mlu, sizeof(int) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
    if (distances) {
      cnrtMemcpy(distances, topk_distances_mlu, sizeof(float) * (size_t)n * k, CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t Flat3::Add(int n, const float *x, const int *ids) {
  LOGT(Flat) << "Add(" << n << ", " << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !x) {
    LOGE(Flat) << "Add() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (((size_t)ntotal_ + n) > std::numeric_limits<int>::max()) {
    LOGE(Flat) << "Add() vectors number to be added over int_max(" << std::numeric_limits<int>::max() << ")";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  if (ntotal_ > 0) {
    if (!ids_.empty() && !ids) {
      LOGE(Flat) << "Add() need ids";
      return CNINDEX_RET_BAD_PARAMS;
    } else if (ids_.empty() && ids) {
      LOGW(Flat) << "Add() index as id, discard input ids";
    }
  }

  int alloc_size = CeilPower2(ntotal_ + n);
  if (alloc_size > nallocated_) {
    void *vectors_base = AllocMLUMemory(sizeof(float) * (size_t)alloc_size * d_);
    if (!vectors_base) return CNINDEX_RET_ALLOC_FAILED;
    if (vectors_base_mlu_ && ntotal_ > 0) {
      cnrtMemcpy(vectors_base, vectors_base_mlu_, sizeof(float) * (size_t)ntotal_ * d_, CNRT_MEM_TRANS_DIR_DEV2DEV);
    }
    vectors_base_mlu_ = vectors_base;
    nallocated_ = alloc_size;
  }

  float *xb_add_mlu = static_cast<float *>(vectors_base_mlu_) + (size_t)ntotal_ * d_;
  cnrtMemcpy(xb_add_mlu, const_cast<float *>(x), sizeof(float) * (size_t)n * d_, CNRT_MEM_TRANS_DIR_HOST2DEV);

  if (ids && (ntotal_ == 0 || !ids_.empty())) {
    ids_.insert(ids_.end(), ids, ids + n);
  }

  ntotal_ += n;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t Flat3::Remove(int n, const int *ids) {
  LOGT(Flat) << "Remove(" << n << ", " << static_cast<const void *>(ids) << ")";

  if (n <= 0 || !ids) {
    LOGE(Flat) << "Remove() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (ntotal_ == 0) {
    LOGW(Flat) << "Remove() no vectors";
    return CNINDEX_RET_SUCCESS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  DeviceGuard(device_id_);

  int vector_size = sizeof(float) * d_;
  if (ids_.empty()) {
    std::vector<int> indices(ids, ids + n);
    std::sort(indices.begin(), indices.end(), [](const int &x, const int &y) { return x > y; });
    for (const int &index : indices) {
      if (index >= ntotal_) {
        LOGW(Flat) << "Remove() index: " << index << " is over ntotal: " << ntotal_;
        continue;
      } else {
        float *vectors_base = static_cast<float *>(vectors_base_mlu_);
        if (index != ntotal_ - 1) {
          cnrtMemcpy(vectors_base + (size_t)index * d_, vectors_base + (size_t)(ntotal_ - 1) * d_, vector_size,
                     CNRT_MEM_TRANS_DIR_DEV2DEV);
        }
        ntotal_--;
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      auto it = std::find(ids_.begin(), ids_.end(), ids[i]);
      if (it == ids_.end()) {
        LOGW(Flat) << "Remove() id: " << ids[i] << " is invalid";
        continue;
      } else {
        int index = std::distance(ids_.begin(), it);
        ids_[index] = ids_.back();
        ids_.erase(ids_.end() - 1);
        float *vectors_base = static_cast<float *>(vectors_base_mlu_);
        if (index != ntotal_ - 1) {
          cnrtMemcpy(vectors_base + (size_t)index * d_, vectors_base + (size_t)(ntotal_ - 1) * d_, vector_size,
                     CNRT_MEM_TRANS_DIR_DEV2DEV);
        }
        ntotal_--;
      }
    }
  }

  if (ntotal_ < (nallocated_ / 2) && ntotal_ > 512) {
    int alloc_size = nallocated_ / 2;
    void *vectors_base = AllocMLUMemory(sizeof(float) * (size_t)alloc_size * d_);
    if (!vectors_base) return CNINDEX_RET_ALLOC_FAILED;
    if (vectors_base_mlu_ && ntotal_ > 0) {
      cnrtMemcpy(vectors_base, vectors_base_mlu_, sizeof(float) * (size_t)ntotal_ * d_, CNRT_MEM_TRANS_DIR_DEV2DEV);
      FreeMLUMemory(vectors_base_mlu_);
      vectors_base_mlu_ = vectors_base;
      nallocated_ = alloc_size;
      LOGI(Flat) << "Remove() reduce allocate size to: " << nallocated_;
    } else {
      FreeMLUMemory(vectors_base);
    }
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t Flat3::GetData(float *x, int *ids) const {
  LOGT(Flat) << "GetData(" << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (!x && ntotal_ != 0) {
    LOGE(Flat) << "GetData() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  if (ntotal_ == 0) {
    return CNINDEX_RET_SUCCESS;
  }
  DeviceGuard(device_id_);
  cnrtMemcpy(x, vectors_base_mlu_, sizeof(float) * (size_t)ntotal_ * d_, CNRT_MEM_TRANS_DIR_DEV2HOST);

  if (ids) {
    if (ids_.empty()) {
      for (int i = 0; i < ntotal_; i++) ids[i] = i;
    } else {
      memcpy(ids, ids_.data(), sizeof(int) * (size_t)ntotal_);
    }
  }

  return CNINDEX_RET_SUCCESS;
}

#endif  // ENABLE_MLU300

}  // impl

}  // cnindex
