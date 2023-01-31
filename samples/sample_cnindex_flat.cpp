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
  int device_id = 0;

  // prepare random data
  size_t n_add = 1024;
  std::vector<float>add_vector(n_add * d);
  std::vector<int>add_ids(n_add);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c1(-1.0, 1.0);
  for (size_t i = 0; i < n_add * d; ++i) {
    add_vector[i] = c1(e);
  }
  for (size_t i = 0; i < n_add; i++) {
    add_ids[i] = i;
  }

  // create cnindex flat
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, device_id);

  // add vector
  {
    int ret = flat.Add((int)n_add, add_vector.data(), add_ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "flat.Add() failed" << std::endl;
      return false;
    } else {
      std::cout << "flat.Add() sucess" << std::endl;
    }
  }

  // search vector
  {
    size_t n_search = n_add / 2;
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

    // search result
    float* distances = new float[n_search * topk];
    std::unique_ptr<float> dmup(distances);
    int *labels = new int[n_search * topk];
    std::unique_ptr<int> lmup(labels);

    int status = flat.Search((int)n_search, vectors.data(), topk, labels, distances);
    if (status != CNINDEX_RET_SUCCESS) {
      std::cout << "cnindex FlatSearch failed" << std::endl;
    } else {
      std::cout << "cnindex FlatSearch sucess" << std::endl;
    }
  }

  // remove vector
  {
    size_t n_remove = n_add / 2;
    std::vector<int> remove_vecotr(n_remove);
    for (size_t i = 0; i < n_remove; i++) {
      remove_vecotr[i] = i;
    }

    int status = flat.Remove((int)n_remove, remove_vecotr.data());
    int size_after_remove = flat.GetSize();

    if (status != CNINDEX_RET_SUCCESS && size_after_remove != n_add - n_remove) {
      std::cout << "cnindex FlatRemove failed" << std::endl;
    } else {
      std::cout << "cnindex FlatRemove sucess! ntotal = " << n_add << ", remove = " << n_remove 
                << ", after remove = " << size_after_remove << std::endl;
    }
  }

  // reset
  {
    int status = flat.Reset();
    int size_after_reset = flat.GetSize();

    if (status != CNINDEX_RET_SUCCESS && size_after_reset != 0) {
      std::cout << "cnindex FlatReset failed" << std::endl;
    } else {
      std::cout << "cnindex FlatReset sucess! after reset size = " << size_after_reset << std::endl;
    }
  }
}
