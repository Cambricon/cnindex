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

#ifndef __CNINDEX_FLAT_BASE_H__
#define __CNINDEX_FLAT_BASE_H__

#include "cnindex.h"

namespace cnindex {

namespace impl {

class Flat {
 public:
  Flat(int d, cnindexMetric_t metric) : d_(d), metric_(metric), ntotal_(0) {}
  virtual ~Flat() {}

  virtual cnindexReturn_t Reset() = 0;

  virtual cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances,
                                 bool output_on_mlu = false) const = 0;
  virtual cnindexReturn_t Add(int n, const float *x, const int *ids) = 0;
  virtual cnindexReturn_t Remove(int n, const int *ids) = 0;

  virtual int GetDimension() const { return d_; }
  virtual int GetSize() const { return ntotal_; }

  virtual cnindexReturn_t GetData(float *x, int *ids) const = 0;
 
  virtual const float * GetDataPointer() const { return nullptr; };
  virtual bool IsCPUImpl() const { return true; };

 protected:
  const int d_;
  const cnindexMetric_t metric_;
  int ntotal_;
};  // Flat

}  // impl

}  // cnindex

#endif  // __CNINDEX_FLAT_BASE_H__
