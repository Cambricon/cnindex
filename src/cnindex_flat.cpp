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
#include "cnindex_flat.h"
#include "cnindex_flat_cpu.h"
#include "cnindex_flat_mlu300.h"

namespace cnindex {

Flat::Flat(int d, cnindexMetric_t metric, int device_id) {
  if (device_id < 0) {
    flat_ = new (std::nothrow) cnindex::impl::FlatCPU(d, metric);
    return;
  }

#ifdef ENABLE_MLU300
  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (device_id >= dev_num) {
    LOGE(Flat) << "Flat() invalid device id: " << device_id;
    return;
  }

  std::string device_name;
#if CNRT_MAJOR_VERSION < 5
  cnrtDeviceInfo_t dev_info;
  cnrtRet_t ret = cnrtGetDeviceInfo(&dev_info, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(Flat) << "Flat() cnrtGetDeviceInfo failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_info.device_name);
#else
  cnrtDeviceProp_t dev_prop;
  cnrtRet_t ret = cnrtGetDeviceProperties(&dev_prop, device_id);
  if (CNRT_RET_SUCCESS != ret) {
    LOGE(Flat) << "Flat() cnrtGetDeviceProperties failed, ret=" << ret;
    return;
  }
  device_name = std::string(dev_prop.name);
#endif
  if (std::string::npos != device_name.find("MLU370")) {
    flat_ = new (std::nothrow) cnindex::impl::Flat3(d, metric, device_id);
  } else
#endif
  flat_ = new (std::nothrow) cnindex::impl::FlatCPU(d, metric);
}

Flat::~Flat() {
  if (flat_) {
    delete flat_;
    flat_ = nullptr;
  }
}

Flat::Flat(Flat &&flat) {
  flat_ = flat.flat_;
  flat.flat_ = nullptr;
}

Flat & Flat::operator=(Flat &&flat) {
  if (flat_) {
    delete flat_;
  }
  flat_ = flat.flat_;
  flat.flat_ = nullptr;
  return *this;
}

cnindexReturn_t Flat::Reset() {
  if (flat_) return flat_->Reset();
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t Flat::Search(int n, const float *x, int k, int *ids, float *distances) const {
  if (flat_) return flat_->Search(n, x, k, ids, distances);
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t Flat::Add(int n, const float *x, const int *ids) {
  if (flat_) return flat_->Add(n, x, ids);
  return CNINDEX_RET_NOT_VALID;
}
cnindexReturn_t Flat::Remove(int n, const int *ids) {
  if (flat_) return flat_->Remove(n, ids);
  return CNINDEX_RET_NOT_VALID;
}

int Flat::GetDimension() const {
  if (flat_) return flat_->GetDimension();
  return CNINDEX_RET_NOT_VALID;
}

int Flat::GetSize() const {
  if (flat_) return flat_->GetSize();
  return CNINDEX_RET_NOT_VALID;
}

cnindexReturn_t Flat::GetData(float *x, int *ids) const {
  if (flat_) return flat_->GetData(x, ids);
  return CNINDEX_RET_NOT_VALID;
}

}  // cnindex
