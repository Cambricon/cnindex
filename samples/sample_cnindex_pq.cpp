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

#include <cnindex.h>

int main(int argc, char* argv[]) {
  int d = 256;
  int M = 32;
  int nbits = 8;
  int device_id = 0;
  int ksub = 1 << nbits;
  int dsub = d / M;
  int code_size = (nbits * M + 7) / 8;

  size_t ntotal = 1000000;
  size_t n_add = 100000;

  // prepare random centroids codes ids
  std::vector<float> centroids(ksub * d);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (int i = 0; i < ksub * d; ++i) {
    centroids[i] = c(e);
  }

  // prepare add vectors and ids
  std::vector<float> addvecs(n_add * d);
  std::vector<int> ids(n_add);
  for (size_t i = 0; i < n_add; i++) {
    for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
    ids[i] = i;
  }

  // create cnindex pq
  cnindex::PQ pq(d, CNINDEX_METRIC_L2, M, nbits, device_id);
  pq.SetCentroids(centroids.data());

  // add vector
  {
    auto status = pq.Add((int)n_add, addvecs.data(), ids.data());
    if (status != CNINDEX_RET_SUCCESS) {
      std::cout << "pq.Add() failed" << std::endl;
      return false;
    } else {
      std::cout << "pq.Add() sucess" << std::endl;
    }
  }

  // search vector
  {
    size_t n_search = 1024;
    int topk = n_add < 32 ? 1 : 32;

    // prepare search vectors
    std::vector<float> vectors(n_search * d);
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> q(-3.0, 3.0);
    for (size_t i = 0; i < n_search; i++) {
      for (int j = 0; j < d; j++) {
        vectors[i * d + j] = q(e);
      }
    }

    int *labels = new int[n_search * topk];
    std::unique_ptr<int> lmup(labels);
    float* distances = new float[n_search * topk];
    std::unique_ptr<float> dmup(distances);

    auto status = pq.Search((int)n_search, vectors.data(), topk, labels, distances);
    if (status != CNINDEX_RET_SUCCESS) {
      std::cout << "pq.Search failed" << std::endl;
    } else {
      std::cout << "pq.Search sucess" << std::endl;
    }
  }

  // remove vector
  {
    size_t n_remove = n_add / 2;
    std::vector<int> remove_vecotr(n_remove);
    for (size_t i = 0; i < n_remove; i++) {
      remove_vecotr[i] = i;
    }

    int status = pq.Remove((int)n_remove, remove_vecotr.data());
    int size_after_remove = pq.GetSize();

    if (status != CNINDEX_RET_SUCCESS && size_after_remove != n_add - n_remove) {
      std::cout << "pq.Remove failed" << std::endl;
    } else {
      std::cout << "pq.Remove sucess! ntotal = " << n_add << " ,remove = " << n_remove 
                << ", after remove = " << size_after_remove << std::endl;
    }
  }

  // reset
  {
    int status = pq.Reset();
    int size_after_reset = pq.GetSize();

    if (status != CNINDEX_RET_SUCCESS && size_after_reset != 0) {
      std::cout << "pq.Reset failed" << std::endl;
    } else {
      std::cout << "pq.Reset sucess! after reset size = " << size_after_reset << std::endl;
    }
  }
}
