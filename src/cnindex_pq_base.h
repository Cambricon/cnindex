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

#ifndef __CNINDEX_PQ_BASE_H__
#define __CNINDEX_PQ_BASE_H__

#include <stdint.h>

#include "cnindex.h"

namespace cnindex {

namespace impl {

class PQ {
 public:
  PQ(int d, cnindexMetric_t metric, int M, int nbits, int device_id)
     : d_(d), metric_(metric), M_(M), nbits_(nbits), code_size_((nbits * M + 7) / 8),
       dsub_(d / M), ksub_(1 << nbits), ntotal_(0), device_id_(device_id) {}
  virtual ~PQ() {}
 
  virtual cnindexReturn_t Reset() = 0;

  virtual cnindexReturn_t SetCentroids(const float *centroids) = 0;
  virtual cnindexReturn_t SetData(int size, const uint8_t *codes, const int *ids) = 0;
  virtual int GetSize() const = 0;
  virtual cnindexReturn_t GetData(uint8_t *codes, int *ids) const = 0;

  virtual cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances) const = 0;
  virtual cnindexReturn_t Add(int n, const float *x, const int *ids) = 0;
  virtual cnindexReturn_t Remove(int n, const int *ids) = 0;

 protected:
  const int d_;
  const cnindexMetric_t metric_;
  const int M_;
  const int nbits_;
  const int code_size_;
  int dsub_;
  int ksub_;

  int ntotal_;

  const int device_id_;
};  // PQ

}  // impl

}  // cnindex

#endif  // __CNINDEX_PQ_BASE_H__
