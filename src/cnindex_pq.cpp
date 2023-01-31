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

#include <cnrt.h>

#include "utils/log.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_pq.h"
#include "cnindex_pq_mlu200.h"
#include "cnindex_pq_mlu300.h"

namespace cnindex {

PQ::PQ(int d, cnindexMetric_t metric, int M, int nbits, int device_id) {
  if (metric != CNINDEX_METRIC_L2) {
    LOGE(PQ) << "PQ() unsupported metric type: " << metric;
    return;
  }

  CNRTInit();

  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (device_id < 0 || device_id >= dev_num) {
    LOGE(PQ) << "PQ() invalid device id: " << device_id;
    return;
  }

  std::string device_name;
#if CNRT_MAJOR_VERSION < 5
  cnrtDeviceInfo_t dev_info;
  cnrtRet_t ret = cnrtGetDeviceInfo(&dev_info, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(PQ) << "PQ() cnrtGetDeviceInfo failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_info.device_name);
#else
  cnrtDeviceProp_t dev_prop;
  cnrtRet_t ret = cnrtGetDeviceProperties(&dev_prop, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(PQ) << "PQ() cnrtGetDeviceProperties failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_prop.name);
#endif

  if (std::string::npos != device_name.find("MLU270")) {
#ifdef ENABLE_MLU200    
    pq_ = new (std::nothrow) cnindex::impl::PQ2(d, metric, M, nbits, device_id);
#endif
  } else if (std::string::npos != device_name.find("MLU370")) {
#ifdef ENABLE_MLU300    
    pq_ = new (std::nothrow) cnindex::impl::PQ3(d, metric, M, nbits, device_id);
#endif    
  } else {
    LOGE(PQ) << "PQ() unsupported MLU device: " << device_name;
    return;
  }

  if (!pq_) {
    LOGE(PQ) << "PQ() find device name: " << device_name 
             << ", but can't init cnindex::impl::PQ" << (device_name.find("MLU270") ? "2" : "3");
    return;
  }
}

PQ::~PQ() {
  if (pq_) {
    delete pq_;
    pq_ = nullptr;
  }
}

PQ::PQ(PQ &&PQ) {
  pq_ = PQ.pq_;
  PQ.pq_ = nullptr;
}

PQ & PQ::operator=(PQ &&PQ) {
  if (pq_) {
    delete pq_;
  }
  pq_ = PQ.pq_;
  PQ.pq_ = nullptr;
  return *this;
}

cnindexReturn_t PQ::Reset() {
  if (pq_) return pq_->Reset();
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::SetCentroids(const float *centroids) {
  if (pq_) return pq_->SetCentroids(centroids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::SetData(int size, const uint8_t *codes, const int *ids) {
  if (pq_) return pq_->SetData(size, codes, ids);
  return CNINDEX_RET_NOT_VALID;
}

int PQ::GetSize() const {
  if (pq_) return pq_->GetSize();
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::GetData(uint8_t *codes, int *ids) const {
  if (pq_) return pq_->GetData(codes, ids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::Search(int n, const float *x, int k, int *ids, float *distances) const {
  if (pq_) return pq_->Search(n, x, k, ids, distances);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::Add(int n, const float *x, const int *ids) {
  if (pq_) return pq_->Add(n, x, ids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t PQ::Remove(int n, const int *ids) {
  if (pq_) return pq_->Remove(n, ids);
  return CNINDEX_RET_NOT_VALID;
}

}  // cnindex
