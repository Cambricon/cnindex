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

#include <cnrt.h>

#include "utils/log.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_ivf.h"
#include "cnindex_ivfpq.h"
#include "cnindex_ivfpq_mlu200.h"
#include "cnindex_ivfpq_mlu300.h"

namespace cnindex {

IVFPQ::IVFPQ(const Flat *flat, cnindexMetric_t metric, int M, int nbits, int device_id) {
  if (metric != CNINDEX_METRIC_L2) {
    LOGE(IVFPQ) << "IVFPQ() unsupported metric type: " << metric;
    return;
  }

  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (device_id < 0 || device_id >= dev_num) {
    LOGE(IVFPQ) << "IVFPQ() invalid device id: " << device_id;
    return;
  }

  std::string device_name;
#if CNRT_MAJOR_VERSION < 5
  cnrtDeviceInfo_t dev_info;
  cnrtRet_t ret = cnrtGetDeviceInfo(&dev_info, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(IVFPQ) << "IVFPQ() cnrtGetDeviceInfo failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_info.device_name);
#else
  cnrtDeviceProp_t dev_prop;
  cnrtRet_t ret = cnrtGetDeviceProperties(&dev_prop, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(IVFPQ) << "IVFPQ() cnrtGetDeviceProperties failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_prop.name);
#endif

  if (std::string::npos != device_name.find("MLU270")) {
#ifdef ENABLE_MLU200    
    ivfpq_ = new (std::nothrow) cnindex::impl::IVFPQ2(flat, metric, M, nbits, device_id);
#endif
  } else if (std::string::npos != device_name.find("MLU370")) {
#ifdef ENABLE_MLU300    
    ivfpq_ = new (std::nothrow) cnindex::impl::IVFPQ3(flat, metric, M, nbits, device_id);
#endif  
  } else {
    LOGE(IVFPQ) << "IVFPQ() unsupported MLU device: " << device_name;
    return;
  }

  if (!ivfpq_) {
    LOGE(IVFPQ) << "IVFPQ() find device name: " << device_name 
                << ", but can't init cnindex::impl::IVFPQ" << (device_name.find("MLU270") ? "2" : "3");
    return;
  }
}

IVFPQ::~IVFPQ() {
  if (ivfpq_) {
    delete ivfpq_;
    ivfpq_ = nullptr;
  }
}

IVFPQ::IVFPQ(IVFPQ &&ivfpq) {
  ivfpq_ = ivfpq.ivfpq_;
  ivfpq.ivfpq_ = nullptr;
}

IVFPQ & IVFPQ::operator=(IVFPQ &&ivfpq) {
  if (ivfpq_) {
    delete ivfpq_;
  }
  ivfpq_ = ivfpq.ivfpq_;
  ivfpq.ivfpq_ = nullptr;
  return *this;
}

cnindexReturn_t IVFPQ::Reset() {
  if (ivfpq_) return ivfpq_->Reset();
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::SetCentroids(const float *centroids) {
  if (ivfpq_) return ivfpq_->SetCentroids(centroids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::SetListData(int index, int size, const uint8_t *codes, const int *ids) {
  if (ivfpq_) return ivfpq_->SetListData(index, size, static_cast<const void *>(codes), ids);
  return CNINDEX_RET_NOT_VALID;
}

int IVFPQ::GetListSize(int index) const {
  if (ivfpq_) return ivfpq_->GetListSize(index);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::GetListData(int index, uint8_t *codes, int *ids) const {
  if (ivfpq_) return ivfpq_->GetListData(index, static_cast<void *>(codes), ids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const {
  if (ivfpq_) return ivfpq_->Search(n, x, nprobe, k, ids, distances);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::Add(int n, const float *x, const int *ids) {
  if (ivfpq_) return ivfpq_->Add(n, x, ids);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t IVFPQ::Remove(int n, const int *ids) {
  if (ivfpq_) return ivfpq_->Remove(n, ids);
  return CNINDEX_RET_NOT_VALID;
}

}  // cnindex
