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

#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <chrono>

#include "cnindex.h"

int main(int argc, char* argv[]) {
  int nlist = 1024;
  int d = 256;
  int M = 32;
  int nbits = 8;
  int ksub = 1 << nbits;

  // prepare random centroids
  std::vector<float>level1_centroids(nlist * d);
  std::vector<float> level2_centroids(d * ksub);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c1(-1.0, 1.0);
  std::uniform_real_distribution<float> c2(-2.0, 2.0);
  for (int i = 0; i < nlist * d; ++i) {
    level1_centroids[i] = c1(e);
  }
  for (int i = 0; i < d * ksub; ++i) {
    level2_centroids[i] = c2(e);
  }

  // create cnindex ivfpq
  int device_id = 0;
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, device_id);
  int ret = flat.Add(nlist, level1_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "flat.Add() failed" << std::endl;
    return false;
  }
  cnindex::IVFPQ ivfpq(&flat, CNINDEX_METRIC_L2, M, nbits, device_id);
  ret = ivfpq.SetCentroids(level2_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.SetCentroids() failed" << std::endl;
    return false;
  }

  // cnindex add
  size_t num_add = 100000;
  {
    // prepare add vectors and ids
    std::vector<float> vectors(num_add * d);
    std::vector<int> ids(num_add * d);
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> q(-3.0, 3.0);
    for (size_t i = 0; i < num_add; ++i) {
      for (int j = 0; j < d; ++j) vectors[i * d + j] = q(e);
      ids[i] = i;
    }

    cnindexReturn_t status = ivfpq.Add((int)num_add, vectors.data(), ids.data());
    if (status != CNINDEX_RET_SUCCESS) {
      std::cout << "cnindex IVFPQAdd failed" << std::endl;
    } else {
      std::cout << "cnindex IVFPQAdd sucess" << std::endl;
    }
  }

  // cnindex search
  {
    size_t num_search = 100;
    int nprobe = 128;
    int topk = 32;
    int dsub = d / M;
    int code_size = (nbits * M + 7) / 8;
    // prepare search vectors
    std::vector<float> vectors(num_search * d);
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> q(-3.0, 3.0);
    for (size_t i = 0; i < num_search; i++) {
      for (int j = 0; j < d; j++) {
        vectors[i * d + j] = q(e);
      }
    }

    // search result
    float* distances = new float[num_search * topk];
    std::unique_ptr<float> dmup(distances);
    int *labels = new int[num_search * topk];
    std::unique_ptr<int> lmup(labels);

    cnindexReturn_t status = ivfpq.Search((int)num_search, vectors.data(), nprobe, topk,
                                          labels, distances);
    if (status != CNINDEX_RET_SUCCESS) {
      std::cout << "cnindex IVFPQSearch failed" << std::endl;
    } else {
      std::cout << "cnindex IVFPQSearch sucess" << std::endl;
    }
  }

  // cnindex remove
  {
    size_t num_remove = 100;
    // prepare remove ids
    std::vector<int> remove_ids;
    for (size_t i = 0; i < num_remove; ++i) {
      remove_ids.push_back(i);
    }

    cnindexReturn_t status = ivfpq.Remove((int)num_remove, remove_ids.data());
    
    int size_after_remove = 0;
    for (int i = 0; i < nlist; i++) {
      size_after_remove += ivfpq.GetListSize(i);
    }

    if (status != CNINDEX_RET_SUCCESS && size_after_remove != num_add - num_remove) {
      std::cout << "cnindex IVFPQRemove failed" << std::endl;
    } else {
      std::cout << "cnindex IVFPQRemove sucess" << std::endl;
    }
  }

  // cnindex reset
  {
    cnindexReturn_t status = ivfpq.Reset();

    int size_after_reset = 0;
    for (int i = 0; i < nlist; i++) {
      size_after_reset += ivfpq.GetListSize(i);
    }

    if (status != CNINDEX_RET_SUCCESS && size_after_reset != 0) {
      std::cout << "cnindex IVFPQReset failed" << std::endl;
    } else {
      std::cout << "cnindex IVFPQReset sucess" << std::endl;
    }
  }
}
