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

#include "utils/distances.h"
#include "utils/log.h"
#include "utils/thread_pool.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_flat_cpu.h"

using cnindex::GetCPUCoreNumber;
using cnindex::GetThreadPool;
using cnindex::EqualityThreadPool;
using cnindex::fvec_L2sqr;
using cnindex::fvec_inner_product;

namespace cnindex {

namespace impl {

#ifdef USE_THREAD_POOL
std::atomic<int> FlatCPU::instances_number_{0};
#endif

FlatCPU::FlatCPU(int d, cnindexMetric_t metric) : Flat(d, metric) {
  if (metric_ != CNINDEX_METRIC_L2 && metric_ != CNINDEX_METRIC_IP) {
    LOGE(Flat) << "Flat() unsupported metric type: " << metric_;
    return;
  }

  LOGC(Flat) << "\033[35mFlatCPU::FlatCPU()\033[0m";

  ntotal_ = 0;
 
  thread_pool_ = cnindex::GetThreadPool();
  size_t thread_pool_size = thread_pool_->Size();
  size_t threads_number = thread_pool_size + parallelism_;
  threads_number = std::min(threads_number, GetCPUCoreNumber());
  if (threads_number > thread_pool_size) thread_pool_->Resize(threads_number);
  instances_number_++;
}

FlatCPU::~FlatCPU() {
  if (exit_) return;
  exit_ = true;

  std::lock_guard<std::mutex> lk(mutex_);

  xb_.clear();
  ids_.clear();

  ntotal_ = 0;

  int threads_number = parallelism_ * --instances_number_;
  if (threads_number < GetCPUCoreNumber()) thread_pool_->Resize(threads_number);

  LOGC(Flat) << "\033[35mFlatCPU::~FlatCPU()\033[0m";
}

cnindexReturn_t FlatCPU::Reset() {
  LOGT(Flat) << "Reset()";

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  xb_.clear();
  ids_.clear();

  ntotal_ = 0;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t FlatCPU::Search(int n, const float *x, int k, int *ids, float *distances, bool output_on_mlu) const {
  LOGT(Flat) << "Search(" << n << ", " << static_cast<const void *>(x) << ", " << k << ", "
             << static_cast<void *>(ids) << ", " << static_cast<void *>(distances) << ")";

  if (n <= 0 || !x || k <= 0 || !ids) {
    LOGE(Flat) << "Search() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (ntotal_ == 0) {
    LOGE(Flat) << "Search() database is empty";
    return CNINDEX_RET_NOT_VALID;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  std::vector<int> indices(n * k);
  int topk = std::min(k, ntotal_);

  using IdxDis = std::pair<int, float>;
  std::vector<IdxDis> idx_dis;
  for (size_t i = 0; i < (size_t)n; i++) {
#ifdef USE_THREAD_POOL
    if (ntotal_ >= (parallelism_ * 64)) {
      auto compute_distances = [this, &idx_dis, k](const float *x, int s, int n) {
        for (int i = 0; i < n; i++) {
          if (exit_) return;
          int index = s + i;
          float distance;
          if (metric_ == CNINDEX_METRIC_L2) {
            distance = fvec_L2sqr(x, xb_.data() + index * d_, d_);
          } else if (metric_ == CNINDEX_METRIC_IP) {
            distance = fvec_inner_product(x, xb_.data() + index * d_, d_);
          }
          // if (k == 1) distance = std::sqrt(distance);
          idx_dis[index] = std::make_pair(index, distance);
        }
      };

      idx_dis.resize(ntotal_);
      std::vector<std::future<void>> fs;
      int bs = ntotal_ / parallelism_;
      int br = ntotal_ % parallelism_;
      int offset = 0;
      for (int j = 0; j < parallelism_; j++) {
        int bn = j < br ? (bs + 1) : bs;
        fs.emplace_back(thread_pool_->Push(compute_distances, x + i * d_, offset, bn));
        offset += bn;
      }
      for (auto &f : fs) f.get();
      if (exit_) return CNINDEX_RET_NOT_VALID;
    } else
#endif
    for (size_t j = 0; j < (size_t)ntotal_; j++) {
      if (exit_) return CNINDEX_RET_NOT_VALID;
      float distance;
      if (metric_ == CNINDEX_METRIC_L2) {
        distance = fvec_L2sqr(x + i * d_, xb_.data() + j * d_, d_);
      } else if (metric_ == CNINDEX_METRIC_IP) {
        distance = fvec_inner_product(x + i * d_, xb_.data() + j * d_, d_);
      }
      idx_dis.emplace_back(j, distance);
    }

    if (ntotal_ == 1) {
      indices[i] = idx_dis[0].first;
      if (distances) distances[i] = idx_dis[0].second;
    } else {
      auto compare_ascend = [](const IdxDis &x, const IdxDis &y) -> bool { return x.second < y.second; };
      auto compare_descend = [](const IdxDis &x, const IdxDis &y) -> bool { return x.second > y.second; };
      if (k == 1) {
        auto top1 = *std::min_element(idx_dis.begin(), idx_dis.end(),
                                      metric_ == CNINDEX_METRIC_L2 ? compare_ascend : compare_descend);
        indices[i] = top1.first;
        if (distances) distances[i] = top1.second;
      } else {
        std::partial_sort(idx_dis.begin(), idx_dis.begin() + topk, idx_dis.end(),
                          metric_ == CNINDEX_METRIC_L2 ? compare_ascend : compare_descend);
        for (int j = 0; j < k; j++) {
          indices[i * k + j] = j < topk ? idx_dis[j].first : -1;
          if (distances) distances[i * k + j] = j < topk ? idx_dis[j].second : std::numeric_limits<float>::max();
        }
      }
    }

    idx_dis.clear();
  }

  for (size_t i = 0; i < (size_t)n * k; i++) {
    ids[i] = indices[i] == -1 ? -1 : (ids_.empty() ? indices[i] : ids_[indices[i]]);
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t FlatCPU::Add(int n, const float *x, const int *ids) {
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

  if (ntotal_ > 0) {
    if (!ids_.empty() && !ids) {
      LOGE(Flat) << "Add() need ids";
      return CNINDEX_RET_BAD_PARAMS;
    } else if (ids_.empty() && ids) {
      LOGW(Flat) << "Add() index as id, discard input ids";
    }
  }

  xb_.insert(xb_.end(), x, x + (size_t)n * d_);

  if (ids && (ntotal_ == 0 || !ids_.empty())) {
    ids_.insert(ids_.end(), ids, ids + n);
  }

  ntotal_ += n;

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t FlatCPU::Remove(int n, const int *ids) {
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

  if (ids_.empty()) {
    std::vector<int> indices(ids, ids + n);
    std::sort(indices.begin(), indices.end(), [](const int &x, const int &y) { return x > y; });
    for (const int &index : indices) {
      if (index >= ntotal_) {
        LOGW(Flat) << "Remove() index: " << index << " is over ntotal: " << ntotal_;
        continue;
      } else {
        memcpy(xb_.data() + (size_t)index * d_, xb_.data() + (size_t)(ntotal_ - 1) * d_, sizeof(float) * d_);
        xb_.erase(xb_.begin() + (size_t)(ntotal_ - 1) * d_, xb_.end());
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
        memcpy(xb_.data() + (size_t)index * d_, xb_.data() + (size_t)(ntotal_ - 1) * d_, sizeof(float) * d_);
        xb_.erase(xb_.begin() + (size_t)(ntotal_ - 1) * d_, xb_.end());
        ntotal_--;
      }
    }
  }

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t FlatCPU::GetData(float *x, int *ids) const {
  LOGT(Flat) << "GetData(" << static_cast<const void *>(x) << ", " << static_cast<const void *>(ids) << ")";

  if (!x) {
    LOGE(Flat) << "GetData() invalid parameters";
    return CNINDEX_RET_BAD_PARAMS;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  if (exit_) return CNINDEX_RET_NOT_VALID;

  memcpy(x, xb_.data(), sizeof(float) * (size_t)ntotal_ * d_);

  if (ids) {
    if (ids_.empty()) {
      for (int i = 0; i < ntotal_; i++) ids[i] = i;
    } else {
      memcpy(ids, ids_.data(), sizeof(int) * (size_t)ntotal_);
    }
  }

  return CNINDEX_RET_SUCCESS;
}

}  // impl

}  // cnindex
