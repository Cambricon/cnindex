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

#include <vector>

#include <cnrt.h>

#include "utils/log.h"
#include "utils/utils.h"

#include "cnindex.h"
#include "cnindex_ivf.h"

namespace cnindex {
  
namespace impl {

IVF::IVF(const cnindex::Flat *flat, cnindexMetric_t metric, int vector_size, int device_id)
    : flat_(flat), metric_(metric), d_(flat->GetDimension()), nlist_(flat->GetSize()), vector_size_(vector_size),
      device_id_(device_id) {
  CNRTInit();

  nlist_size_.resize(nlist_, 0);
  vectors_ptr_.resize(nlist_, nullptr);
  ids_ptr_.resize(nlist_, nullptr);
  ntotal_ = 0;
}

IVF::~IVF() {
  for (const auto &p : vectors_ptr_) FreeMLUMemory(p);
  nlist_size_.clear();
  vectors_ptr_.clear();
  ids_ptr_.clear();
}

cnindexReturn_t IVF::Reset() {
  for (const auto &p : vectors_ptr_) FreeMLUMemory(p);
  vectors_ptr_.assign(nlist_, nullptr);
  ids_ptr_.assign(nlist_, nullptr);
}

cnindexReturn_t IVF::SetListData(int index, int size, const void *vectors, const int *ids) {
  if (index >= nlist_ || index < 0 || !vectors || !ids) {
    if (index >= nlist_ || index < 0) {
      LOGE(IVF) << "SetListData() invalid list index: " << index;
    } else {
      LOGE(IVF) << "SetListData() invalid parameters";
    }
    return CNINDEX_RET_BAD_PARAMS;
  }
  if (size <= 0) {
    LOGW(IVF) << "SetListData() list[" << index << "] is empty";
    return CNINDEX_RET_BAD_PARAMS;
  }

  size_t memory_size = ALIGN_128(vector_size_ * size) + ALIGN_128(sizeof(int) * size);

  DeviceGuard(device_id_);
  if (vectors_ptr_[index]) FreeMLUMemory(vectors_ptr_[index]);
  vectors_ptr_[index] = AllocMLUMemory(memory_size);
  cnrtMemcpy(vectors_ptr_[index], const_cast<void *>(vectors), vector_size_ * size, CNRT_MEM_TRANS_DIR_HOST2DEV);
  ids_ptr_[index] = static_cast<uint8_t *>(vectors_ptr_[index]) + ALIGN_128(vector_size_ * size);
  cnrtMemcpy(ids_ptr_[index], const_cast<int *>(ids), sizeof(int) * size, CNRT_MEM_TRANS_DIR_HOST2DEV);
  nlist_size_[index] = size;

  return CNINDEX_RET_SUCCESS;
}

int IVF::GetListSize(int index) const {
  if (index >= nlist_ || index < 0) {
    LOGE(IVF) << "GetListSize() invalid list index: " << index;
    return CNINDEX_RET_BAD_PARAMS;
  }
  return nlist_size_[index];
}

cnindexReturn_t IVF::GetListData(int index, void *vectors, int *ids) const {
  if (index >= nlist_ || index < 0 || !vectors) {
    if (index >= nlist_ || index < 0) {
      LOGE(IVF) << "GetListData() invalid list index: " << index;
    } else {
      LOGE(IVF) << "GetListData() invalid parameters";
    }
    return CNINDEX_RET_BAD_PARAMS;
  }

  if (nlist_size_[index] <= 0) return CNINDEX_RET_SUCCESS;

  DeviceGuard(device_id_);
  cnrtMemcpy(vectors, vectors_ptr_[index], vector_size_ * nlist_size_[index], CNRT_MEM_TRANS_DIR_DEV2HOST);
  cnrtMemcpy(ids, ids_ptr_[index], sizeof(int) * nlist_size_[index], CNRT_MEM_TRANS_DIR_DEV2HOST);

  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVF::Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const {
  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVF::Add(int n, const float *x, const int *ids) {
  return CNINDEX_RET_SUCCESS;
}

cnindexReturn_t IVF::Remove(int n, const int *ids) {
  return CNINDEX_RET_SUCCESS;
}

}  // impl

}  // cnindex
