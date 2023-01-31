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

#include <fstream>
#include <random>
#include <chrono>

#include "cnindex.h"
#include "../src/utils/distances.h"
#include "common.cpp"

class CpuPQ {
 public:
  CpuPQ(int d, int M, int nbits): d_(d), M_(M), nbits_(nbits) {
    dsub_ = d_ / M_;
    ksub_ = 1 << nbits_;
    ntotal_ = 0;
    code_size_ = (nbits * M + 7) / 8;
    codes_.clear();
    ids_.clear();}

  ~CpuPQ() {}

  int Add(int n, const float *x, const int *ids);
  int Search(int n, float* x, int topk, int* labels, float* distances);
  int Remove(int n, int* remove_ids);
  void SetData(int size, const uint8_t *codes, const int *ids);
  int Reset() {
    codes_.clear();
    ids_.clear();
    ntotal_ = 0;
  }
  int SetCentroids(const float *centroids) {
    centroids_.resize(ksub_ * d_);
    memcpy(centroids_.data(), centroids, sizeof(float) * ksub_ * d_);
  }

 public:
  int d_;
  int M_;
  int nbits_;
  int code_size_;
  int dsub_;
  int ksub_;
  int ntotal_;
  std::vector<uint8_t> codes_;
  std::vector<int> ids_;
  std::vector<float> centroids_;
};

int CpuPQ::Add(int n, const float *x, const int *ids) {
  if (n <= 0 || !x || !ids) {
    std::cout << "[ERROR]CpuPQ::Add invalid parameters!" << std::endl;
    return -1;
  }

  for (int i = 0; i < n; i++) {
    float distance_min = 0;
    // pq encode
    std::vector<uint8_t> code;
    for (int j = 0; j < M_; j++) {
      distance_min = 0;
      uint8_t idx_min = 0;
      for (int k = 0; k < ksub_; k++) {
        float *centroids_sub_k = centroids_.data() + (j * ksub_ + k) * dsub_;
        float distance = cnindex::fvec_L2sqr(x + i * d_ + j * dsub_, centroids_sub_k, dsub_);
        if (k == 0 || distance < distance_min) {
          distance_min = distance;
          idx_min = k;
        }
      }
      code.push_back(idx_min);
    }

    // insert encoded vector 
    codes_.insert(codes_.end(), code.begin(), code.end());
    ids_.insert(ids_.end(), ids[i]);
    ntotal_++;
  }
}

int CpuPQ::Search(int n, float* x, int topk, int* labels, float* distances) {
  // trans M * ksub * dsub -> ksub * d
  std::vector<float> centroids_trans(ksub_ * d_);
  typedef std::pair<int32_t, float> Pair;
  std::vector<Pair> pq_ids_distances;

  for (int i = 0; i < ksub_; i++) {
    for (int j = 0; j < M_; j++) {
      int boundary = dsub_ * ksub_ * j + i * dsub_;
      int dst_left = i * d_ + dsub_ * j;
      memcpy(centroids_trans.data() + dst_left, centroids_.data() + boundary, dsub_ * sizeof(float));
    }
  }

  for (int i = 0; i < n; i++) {
    std::vector<float> pq_distances;

    // do pq search
    pq_distances.resize(ntotal_);
    pq_search(x + i * d_, centroids_trans.data(), codes_.data(), pq_distances.data(), 1, d_, topk,
              ntotal_, code_size_);
    for (int idx = 0; idx < ntotal_; idx++) {
      pq_ids_distances.emplace_back(ids_[idx], pq_distances[idx]);
    }
    pq_distances.clear();

    int out_size = pq_ids_distances.size();
    int sort_end = std::min(out_size, topk);
    partial_sort(pq_ids_distances.begin(), pq_ids_distances.begin() + sort_end, pq_ids_distances.end(),
                 [](Pair& x, Pair& y) -> bool { return x.second < y.second; });

    // output topk labels and distances
    for (int o = 0; o < topk; o++) {
      labels[o + i * topk] = o < out_size ? pq_ids_distances[o].first : -1;
      distances[o + i * topk] = o < out_size ? pq_ids_distances[o].second : std::numeric_limits<float>::max();
    }

    pq_ids_distances.clear();
  }
}

int CpuPQ::Remove(int n, int* remove_ids) {
  for (int i = 0; i < n; i++) {
    int id = remove_ids[i];
    auto iter = std::find(std::begin(ids_), std::end(ids_), id);
    if(iter == std::end(ids_)) {
      std::cout << "CPU PQ remove id " << id << " failed" << std::endl;
      continue;
    }else{
      int idx = std::distance(std::begin(ids_), iter);
      for (int m_idx = 0; m_idx < M_; m_idx++) {
        codes_[(size_t)idx * M_ + m_idx] = codes_[(size_t)(ntotal_ - 1) * M_ + m_idx];
      }
      codes_.erase(std::begin(codes_) + (size_t)(ntotal_ - 1) * M_, std::begin(codes_) + (size_t)ntotal_ * M_);
      ids_[idx] = ids_[ntotal_ - 1];
      ids_.erase(ids_.begin() + ntotal_ - 1);
      ntotal_--;
    }
  }
}

void CpuPQ::SetData(int size, const uint8_t *codes, const int *ids) {
  codes_.assign(codes, codes + M_ * (size_t)size);
  ids_.assign(ids, ids + size);
  ntotal_ = size;
}

static int g_device_id = 0;

bool test_add(int d, int M, int nbits, int add_num) {
  int ksub = 1 << nbits;

  // prepare random centroids
  std::vector<float> centroids(ksub * d);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (int i = 0; i < ksub * d; ++i) {
    centroids[i] = c(e);
  }

  // create cnindex pq
  cnindex::PQ mlu_pq(d, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  mlu_pq.SetCentroids(centroids.data());

  // create cpu pq
  CpuPQ cpu_pq(d, M, nbits);
  cpu_pq.SetCentroids(centroids.data());

  {
    std::cout << "\n-------------- PQ ADD ACCURACY TEST -----------------" << std::endl;
    std::cout << "ADD dataset:     " << "random"<< std::endl
              << "       add num:  " << add_num << std::endl
              << "       d:        " << d << std::endl
              << "       M:        " << M << std::endl
              << "       nbits:    " << nbits << std::endl << std::endl;

    // prepare add vectors and ids
    std::vector<float> addvecs(add_num * d);
    std::vector<int> ids(add_num);
    for (int i = 0; i < add_num; i++) {
      for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
      ids[i] = i;
    }

    // add
    cpu_pq.Add(add_num, addvecs.data(), ids.data());
    auto ret = mlu_pq.Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "MLU pq.Add() failed" << std::endl;
      return false;
    }

    int m = cpu_pq.M_;
    int ntotal = mlu_pq.GetSize();
    if (ntotal != cpu_pq.ntotal_) {
      std::cout << "ADD failed: mlu ntotal != cpu ntotal" << std::endl;
      return false;
    }

    // compare codes ids
    std::vector<int> mlu_idx(ntotal);
    std::vector<uint8_t> mlu_codes((size_t)ntotal * m);
    std::vector<int> cpu_idx;
    std::vector<uint8_t> cpu_codes;
    mlu_pq.GetData(mlu_codes.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_pq.ids_.data(), cpu_pq.ids_.data() + ntotal);
    cpu_codes.insert(cpu_codes.end(), cpu_pq.codes_.data(), cpu_pq.codes_.data() + (size_t)ntotal * m);

    std::pair<float, float> result = compare_codes_ids(cpu_codes.data(), cpu_idx.data(), mlu_codes.data(), 
                                                       mlu_idx.data(), ntotal, m, true);
    std::cout << "all diff codes: " << (result.first > 0 ? RED : CYAN) << result.first << "%"
            << RESET << ", ids: " << (result.second > 0 ? RED : CYAN) << result.second << "%"
            << RESET << std::endl;
  }

  {
    std::cout << "\n-------------- PQ ADD PERFORMANCE TEST --------------" << std::endl;
    std::cout << "Add dataset:    " << "ramdom" << std::endl
              << "    nadd:       " << add_num << std::endl
              << "    d:          " << d << std::endl
              << "    M:          " << M << std::endl
              << "    nbits:      " << nbits << std::endl << std::endl;

    mlu_pq.Reset();
    // prepare add vectors and ids
    std::vector<float> addvecs(add_num * d);
    std::vector<int> ids(add_num);
    for (int i = 0; i < add_num; i++) {
      for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
      ids[i] = i;
    }

    int warm_iter_num = 1;
    for (int i = 0; i < warm_iter_num; i++) {
      auto ret = mlu_pq.Add(1, addvecs.data(), ids.data());
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu pq.Add() failed" << std::endl;
        return false;
      }
    }
    mlu_pq.Reset();

    double add_total_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    auto ret = mlu_pq.Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Add() failed" << std::endl;
      return false;
    }
    std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
    add_total_time += elapsed.count();

    std::cout << CYAN << "MLU Add E2Etime: " << add_total_time / add_num << "us, "
                  << add_num * 1E6 / add_total_time << "qps" << RESET << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
  }

  return true;
}

bool test_search(std::string mode_set, int nq, int d, int M, int nbits, int ntotal, int topk) {
  int ksub = 1 << nbits;
  int dsub = d / M;
  int code_size = (nbits * M + 7) / 8;

  // prepare random centroids codes ids
  std::vector<float> centroids(ksub * d);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (int i = 0; i < ksub * d; ++i) {
    centroids[i] = c(e);
  }
  
  std::vector<uint8_t> codes((size_t)ntotal * code_size);
  std::vector<int> ids(ntotal);
  for (size_t i = 0; i < (size_t)ntotal * code_size; i++) {
    codes[i] = rand() % 256;
  }
  for (int i = 0; i < ntotal; i++) {
    ids[i] = i;
  }

  // create cnindex pq
  cnindex::PQ mlu_pq(d, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  mlu_pq.SetCentroids(centroids.data());

  // create cpu pq
  CpuPQ cpu_pq(d, M, nbits);
  cpu_pq.SetCentroids(centroids.data());

  //set data
  cpu_pq.SetData(ntotal, codes.data(), ids.data());
  mlu_pq.SetData(ntotal, codes.data(), ids.data());

  // search result
  int *labels_cpu = new int[(size_t)nq * topk];
  std::unique_ptr<int> lcup(labels_cpu);
  float *distances_cpu = new float[(size_t)nq * topk];
  std::unique_ptr<float> dcup(distances_cpu);
  int *labels_mlu = new int[(size_t)nq * topk];
  std::unique_ptr<int> lmup(labels_mlu);
  float* distances_mlu = new float[(size_t)nq * topk];
  std::unique_ptr<float> dmup(distances_mlu);

  // query vector
  std::default_random_engine q(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> q_c(-3.0, 3.0);
  std::vector<float> query(nq * d);
  for (int i = 0; i < nq * d; i++) {
    query[i] = q_c(q);
  }

  {
    std::cout << "\n-------------- PQ SEARCH ACCURACY TEST --------------" << std::endl;
    std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
              << "       nquery:   " << nq << std::endl
              << "       d:        " << d << std::endl
              << "       M:        " << M << std::endl
              << "       nbits:    " << nbits << std::endl
              << "       ntotal:   " << ntotal << std::endl
              << "       topk:     " << topk << std::endl << std::endl;

    // search accuracy
    cpu_pq.Search(nq, query.data(), topk, labels_cpu, distances_cpu);
    std::cout << "CPU Search OK" << std::endl;
    auto ret = mlu_pq.Search(nq, query.data(), topk, labels_mlu, distances_mlu);
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Search() failed" << std::endl;
      return false;
    }
    std::cout << "MLU Search OK" << std::endl;

    float labels_diff;
    std::pair<float, float> distances_diff;
    labels_diff = compare(labels_cpu, labels_mlu, nq, topk, true, true);
    std::cout << CYAN << "Diff labels: " << (labels_diff > 0 ? RED : CYAN) << labels_diff * 100 << "%" << std::endl;
    distances_diff = compare_mae_mse(distances_cpu, distances_mlu, nq, topk);
    float distances_mae = distances_diff.first, distances_mse = distances_diff.second;
    std::cout << CYAN << "  distances: " << (distances_mae > 0.01 ? RED : CYAN) << distances_mae * 100 << "%, "
              << (distances_mse > 0.01 ? RED : CYAN) << distances_mse * 100 << "%" << RESET << std::endl;
  }

  {
    std::cout << "\n------------ PQ SEARCH PERFORMANCE TEST -------------" << std::endl;
    std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
              << "       nquery:   " << nq << std::endl
              << "       d:        " << d << std::endl
              << "       M:        " << M << std::endl
              << "       nbits:    " << nbits << std::endl
              << "       ntotal:   " << ntotal << std::endl
              << "       topk:     " << topk << std::endl << std::endl;

    // cpu search
    int cpu_iter_num = 2;
    double cpu_total_time = 0.0;
    for (int i = 0; i < cpu_iter_num; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      cpu_pq.Search(nq, query.data(), topk, labels_cpu, distances_cpu);
      std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
      cpu_total_time += elapsed.count();
    }
    std::cout << "CPU Search OK" << std::endl;

    int warm_iter_num = 1;
    int mlu_iter_num = 2;
    double mlu_total_time = 0.0;
    // mlu warm up
    for (int i = 0; i < warm_iter_num; i++) {
      int ret = mlu_pq.Search(1, query.data(), topk, labels_mlu, distances_mlu);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu pq.Search() failed" << std::endl;
        return false;
      }
    }
    // mlu search
    for (int i = 0; i < mlu_iter_num; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      int ret = mlu_pq.Search(nq, query.data(), topk, labels_mlu, distances_mlu);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu pq.Search() failed" << std::endl;
        return false;
      }
      std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
      mlu_total_time += elapsed.count();
    }
    std::cout << "MLU Search OK" << std::endl;

    std::cout << CYAN << "CPU E2Etime: " << cpu_total_time / cpu_iter_num << "us ("
              << cpu_total_time << "/" << cpu_iter_num << "), "
              << nq * cpu_iter_num * 1E6 / cpu_total_time << "qps" << RESET << std::endl;
    std::cout << CYAN << "MLU E2Etime: " << mlu_total_time / mlu_iter_num << "us ("
              << mlu_total_time << "/" << mlu_iter_num << "), "
              << nq * mlu_iter_num * 1E6 / mlu_total_time << "qps" << RESET << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
  }

  return true;
}

bool test_remove(int nremove, int d, int M, int nbits, int ntotal) {
  int ksub = 1 << nbits;
  int dsub = d / M;
  int code_size = (nbits * M + 7) / 8;

  // prepare random centroids codes ids
  std::vector<float> centroids(ksub * d);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (int i = 0; i < ksub * d; ++i) {
    centroids[i] = c(e);
  }
  
  std::vector<uint8_t> codes((size_t)ntotal * code_size);
  std::vector<int> ids(ntotal);
  for (size_t i = 0; i < (size_t)ntotal * code_size; i++) {
    codes[i] = rand() % 256;
  }
  for (int i = 0; i < ntotal; i++) {
    ids[i] = i;
  }

  // create cnindex pq
  cnindex::PQ mlu_pq(d, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  mlu_pq.SetCentroids(centroids.data());

  // create cpu pq
  CpuPQ cpu_pq(d, M, nbits);
  cpu_pq.SetCentroids(centroids.data());

  //set data
  cpu_pq.SetData(ntotal, codes.data(), ids.data());
  mlu_pq.SetData(ntotal, codes.data(), ids.data());

  // genrate unduplicated remove ids from total ids
  srand(35);
  std::vector<int> total_ids(ids);
  std::vector<int> remove_ids;
  for (int i = 0; i < nremove; i++) {
    int idx = total_ids.size() > 1 ? rand() % (total_ids.size() - 1) : 0;
    remove_ids.push_back(total_ids[idx]);
    total_ids[idx] = total_ids.back();
    total_ids.erase(total_ids.end() - 1);
  }

  {
    std::cout << "\n-------------- PQ REMOVE ACCURACY TEST --------------" << std::endl;
    std::cout << "Remove dataset:    " << "ramdom" << std::endl
              << "       nremove:    " << nremove << std::endl
              << "       d:          " << d << std::endl
              << "       M:          " << M << std::endl
              << "       nbits:      " << nbits << std::endl
              << "       ntotal:     " << ntotal << std::endl << std::endl;

    // Remove
    cpu_pq.Remove(nremove, remove_ids.data());
    std::cout << "CPU Remove OK" << std::endl;
    auto ret = mlu_pq.Remove(nremove, remove_ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Remove() failed" << std::endl;
      return false;
    }
    std::cout << "MLU Remove OK" << std::endl;
    
    // the list_size after remove
    int size_after_remove = mlu_pq.GetSize();
    if (size_after_remove != cpu_pq.ntotal_) {
      std::cout << "CPU and MLU size after remove not equal!" << std::endl;
      return false;
    }
    std::cout << "REMOVE RESULT : "
              << " add num = " << ntotal << "  remove num = " << nremove
              << "  add - remove = " << size_after_remove << std::endl;

    // compare codes ids after remove
    std::vector<int> mlu_idx(size_after_remove);
    std::vector<uint8_t> mlu_codes((size_t)size_after_remove * M);
    std::vector<int> cpu_idx;
    std::vector<uint8_t> cpu_codes;
    mlu_pq.GetData(mlu_codes.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_pq.ids_.data(), cpu_pq.ids_.data() + size_after_remove);
    cpu_codes.insert(cpu_codes.end(), cpu_pq.codes_.data(), cpu_pq.codes_.data() + (size_t)size_after_remove * M);

    std::pair<float, float> result = compare_codes_ids(cpu_codes.data(), cpu_idx.data(), mlu_codes.data(), 
                                                       mlu_idx.data(), size_after_remove, M, true);
    std::cout << "Remove compare diff with codes: " << (result.first > 0 ? RED : CYAN) << result.first << "%"
            << RESET << ", ids: " << (result.second > 0 ? RED : CYAN) << result.second << "%"
            << RESET << std::endl;
  }

  {
    std::cout << "\n------------ PQ REMOVE PERFORMANCE TEST -------------" << std::endl;
    std::cout << "Remove dataset:    " << "ramdom" << std::endl
              << "       nremove:    " << nremove << std::endl
              << "       d:          " << d << std::endl
              << "       M:          " << M << std::endl
              << "       nbits:      " << nbits << std::endl
              << "       ntotal:     " << ntotal << std::endl << std::endl;
    // warm up
    int warm_iter_num = 1;
    auto ret = mlu_pq.Reset();
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Reset() failed" << std::endl;
      return false;
    }
    //set data
    mlu_pq.SetData(ntotal, codes.data(), ids.data());
    for (int i = 0; i < warm_iter_num; i++) {
      mlu_pq.Remove(1, remove_ids.data() + i);
    }

    // mlu remove
    double remove_total_time = 0.0;
    ret = mlu_pq.Reset();
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Reset() failed" << std::endl;
      return false;
    }
    //set data
    mlu_pq.SetData(ntotal, codes.data(), ids.data());

    double test_min_time = std::numeric_limits<double>::max(), test_max_time = 0;
    for (int i = 0; i < nremove; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      ret = mlu_pq.Remove(1, remove_ids.data() + i);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu pq.Remove() failed" << std::endl;
        return false;
      }
      std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
      double test_time = elapsed.count();
      test_min_time = std::min(test_time, test_min_time);
      test_max_time = std::max(test_time, test_max_time);
      remove_total_time += elapsed.count();
    }

    std::cout << CYAN << "MLU Remove E2Etime: min=" << test_min_time << "us, max="
              << test_max_time << "us, average=" << remove_total_time / nremove << "us, "
              << nremove * 1E6 / remove_total_time << "qps\n" << RESET << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
  }

  return true;
}

int main(int argc, char* argv[]) {
  if (!(argc <= 9 && argc >= 8)) {
    printf("[ERROR] ./test_cnindex_pq device_id mode nq d M nbits ntotal topk\n");
    printf("         mode: s: search test using random dataset\n"
           "               \"test-data\": search test using dataset from files\n"
           "               a: add test\n"
           "               r: remove test\n");
    return 0;
  }

#if 1
  g_device_id = atoi(argv[1]);
  std::string mode_set = argv[2];
  int nq = atoi(argv[3]);
  int d = atoi(argv[4]);
  int M = atoi(argv[5]);
  int nbits = atoi(argv[6]);
  int ntotal = atoi(argv[7]);
  int topk = argc == 9 ? atoi(argv[8]) : 1;
#else
  std::string mode_set = "s";
  int nq = 1;
  int d = 256;
  int M = 32;
  int nbits = 8;
  int ntotal = 400;
  int topk = 32;
#endif

  if (mode_set == "a") {
    if (!test_add(d, M, nbits, ntotal)) return -1;
  } else if (mode_set == "r") {
    if (!test_remove(nq, d, M, nbits, ntotal)) return -1;
  } else {
    if (!test_search(mode_set, nq, d, M, nbits, ntotal, topk)) return -1;
  }

  return 0;
}
