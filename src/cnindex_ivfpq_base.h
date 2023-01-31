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

#ifndef __CNINDEX_IVFPQ_BASE_H__
#define __CNINDEX_IVFPQ_BASE_H__

#include <stdint.h>
#include <vector>

#include "cnindex.h"
#include "cnindex_ivf.h"

namespace cnindex {

namespace impl {

class IVFPQ : public IVF {
 public:
  IVFPQ(const cnindex::Flat *flat, cnindexMetric_t metric, int M, int nbits, int device_id)
      : IVF(flat, metric, (nbits * M + 7) / 8, device_id), M_(M), nbits_(nbits), dsub_(d_ / M), ksub_(1 << nbits) {}
  virtual ~IVFPQ() {}

  virtual cnindexReturn_t SetCentroids(const float *centroids) = 0;

 protected:
  int M_;
  int nbits_;
  int dsub_;
  int ksub_;
};  // IVFPQ

}  // impl

}  // cnindex

#endif  // __CNINDEX_IVFPQ_BASE_H__
