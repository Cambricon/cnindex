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
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "cnindex.h"

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define YELLOW  "\033[33m"      /* Yellow */
#define CYAN    "\033[36m"      /* Cyan */

template<typename T>
int read_data(const std::string &path, T *pointer, int count) {
  std::ifstream in(path);
  std::string line;

  if (in) {
    int i = 0;
    T x;
    while (getline(in, line)) {
      std::stringstream ss(line);
      ss >> x;
      pointer[i] = x;
      i++;
      if (count > 0 && i == count)
        break;
    }
    return i;
  } else {
    std::cout << "[READDATA ERROR]No such file: " << path << std::endl;
    return -1;
  }
}

template<typename T>
int read_lib(const std::string &path, T *pointer, int count) {
  std::ifstream in(path);
  std::string line;

  if (in) {
    int i = 0;
    int x;
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      ss >> x;
      pointer[i] = (T)x;
      i++;
      if (i == count)
        break;
    }
    return i;
  } else {
    std::cout << "[READLIB ERROR]No such file: " << path << std::endl;
    return -1;
  }
}

template<class T>
float compare(const T *ref, const T *test, int batch, int size, bool s = false, bool p = false) {
  int diff = 0;
  const T *r = ref, *t = test;
  std::vector<T> rv;
  std::vector<T> tv;
  int total = batch * size;

  if (s) {
    rv.assign(ref, ref + total);
    tv.assign(test, test + total);
    for (int i = 0; i < batch; i++) {  // last item diff && !(last two items exchange)
      if (rv[(i + 1) * size - 1] != tv[(i + 1) * size - 1] &&
          !(rv[(i + 1) * size - 2] == tv[(i + 1) * size - 1] && rv[(i + 1) * size - 1] == tv[(i + 1) * size - 2])) {
        std::sort(rv.begin() + i * size, rv.begin() + (i + 1) * size - 1);
        std::sort(tv.begin() + i * size, tv.begin() + (i + 1) * size - 1);
      } else {
        std::sort(rv.begin() + i * size, rv.begin() + (i + 1) * size);
        std::sort(tv.begin() + i * size, tv.begin() + (i + 1) * size);
      }
    }
    r = rv.data();
    t = tv.data();
  }

  for (int i = 0; i < batch; i++) {
    bool error = false;
    for (int j = 0; j < size; j++) {
      if (t[i * size + j] != r[i * size + j]) {
        diff++;
        error = true;
      }
    }
    if (p && error) {
      std::cout << "\nref[" << i << "]: " << std::endl;
      for (int j = 0; j < size; j++) {
        std::cout << (test[i * size + j] != ref[i * size + j] ? CYAN : RESET) << ref[i * size + j] << " " << RESET;
      }
      std::cout << "\ntest[" << i << "]: " << std::endl;
      for (int j = 0; j < size; j++) {
        std::cout << (test[i * size + j] != ref[i * size + j] ? RED : RESET) << test[i * size + j] << " " << RESET;
      }
      std::cout << "\n";
    }
  }

  return (float)diff / total;
}

template<class T>
std::pair<float, float>
compare_mae_mse(const T *ref, const T *test, int batch, int size) {
  float diff = 0.0;
  float mae = 0.0;
  float mse = 0.0;
  int total = batch * size;

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < size; j++) {
      diff = test[i * size + j] - ref[i * size + j];
      mae += std::abs(diff);
      mse += diff * diff;
    }
  }

  return std::make_pair(mae / total, mse / total);
}

std::pair<float, float>
compare_vectors_ids(const float *ref_vectors, const int *ref_ids, const float *test_vectors, const int *test_ids,
                  int count, int d, bool s = false, bool p = true) {
  typedef std::pair<const float *, int> VectorId;

  int vectors_diff = 0;
  int ids_diff = 0;
  std::vector<VectorId> rv;
  std::vector<VectorId> tv;

  for (int i = 0; i < count; i++) {
    rv.emplace_back(ref_vectors + i * d, ref_ids[i]);
    tv.emplace_back(test_vectors + i * d, test_ids[i]);
  }
  if (s) {
    std::sort(rv.begin(), rv.end(), [](const VectorId &x, const VectorId &y) -> bool { return x.second < y.second; });
    std::sort(tv.begin(), tv.end(), [](const VectorId &x, const VectorId &y) -> bool { return x.second < y.second; });
  }

  for (int i = 0; i < count; i++) {
    if (memcmp(tv[i].first, rv[i].first, sizeof(float) * d)) {
      vectors_diff++;
      if (p) {
        std::cout << "\nref[" << rv[i].second << "]: " << std::endl;
        for (int j = 0; j < d; j++) {
          std::cout << (*(tv[i].first + j) != *(rv[i].first + j) ? CYAN : RESET) << *(rv[i].first + j) << " " << RESET;
        }
        std::cout << "\ntest[" << tv[i].second << "]: " << std::endl;
        for (int j = 0; j < d; j++) {
          std::cout << (*(tv[i].first + j) != *(rv[i].first + j) ? RED : RESET) << *(tv[i].first + j) << " " << RESET;
        }
        std::cout << "\n";
      }
    }
    ids_diff += tv[i].second == rv[i].second ? 0 : 1;
  }

  return std::make_pair((float)vectors_diff / count, (float)ids_diff / count);
}

std::pair<float, float>
compare_codes_ids(const uint8_t *ref_codes, const int *ref_ids, const uint8_t *test_codes, const int *test_ids,
                  int count, int M, bool s = false, bool p = true) {
  typedef std::pair<const uint8_t *, int> CodeId;

  int codes_diff = 0;
  int ids_diff = 0;
  std::vector<CodeId> rv;
  std::vector<CodeId> tv;

  for (int i = 0; i < count; i++) {
    rv.emplace_back(ref_codes + i * M, ref_ids[i]);
    tv.emplace_back(test_codes + i * M, test_ids[i]);
  }
  if (s) {
    std::sort(rv.begin(), rv.end(), [](const CodeId &x, const CodeId &y) -> bool { return x.second < y.second; });
    std::sort(tv.begin(), tv.end(), [](const CodeId &x, const CodeId &y) -> bool { return x.second < y.second; });
  }

  for (int i = 0; i < count; i++) {
    if (memcmp(tv[i].first, rv[i].first, sizeof(uint8_t) * M)) {
      codes_diff++;
      if (p) {
        std::cout << "\nref[" << rv[i].second << "]: " << std::endl;
        for (int j = 0; j < M; j++) {
          std::cout << (*(tv[i].first + j) != *(rv[i].first + j) ? CYAN : RESET) << (int)*(rv[i].first + j)
                    << " " << RESET;
        }
        std::cout << "\ntest[" << tv[i].second << "]: " << std::endl;
        for (int j = 0; j < M; j++) {
          std::cout << (*(tv[i].first + j) != *(rv[i].first + j) ? RED : RESET) << (int)*(tv[i].first + j)
                    << " " << RESET;
        }
        std::cout << "\n";
      }
    }
    ids_diff += tv[i].second == rv[i].second ? 0 : 1;
  }

  return std::make_pair((float)codes_diff / count, (float)ids_diff / count);
}

template <class T>
static float pq_distance(float *x, T *code_vec, float *code_book, int k, int D, int m) {
  if (D % m != 0) {
    std::cout << "[PQDISTANCE ERROR] D % m != 0" << std::endl;
    return 0;
  }
  int Dsub = D / m;
  // k center and m space
  float distance = 0;
  for (int i = 0; i < m; i++) {
    int centroid_index = (int)code_vec[i];
    float *centroid = code_book + centroid_index * D + i * Dsub;
    float *x_seg_i = x + i * Dsub;
    distance += cnindex::fvec_L2sqr(x_seg_i, centroid, Dsub);
  }
  return distance;
};

template <class T>
static void pq_search(float *query_m, float *code_book, const T *code_db, float *out_distance,
               int nq, int D, int k, uint64_t n, int m) {
  T *code_db_j = (T *)malloc(sizeof(T) * m);

  for (size_t j = 0; j < n; j++) {
    for (int o = 0; o < m; o++) {
      code_db_j[o] = code_db[j * m + o];
    }
    out_distance[j] = pq_distance(query_m, code_db_j, code_book, k, D, m);
  }
  free(code_db_j);
};
