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

// #define DUMP_DATA

class CpuFlat {
 public:
  CpuFlat(int d, cnindexMetric_t metric): d_(d), metric_(metric) {
    ntotal_ = 0;
    xb_.clear();
    ids_.clear();
  }

  ~CpuFlat() {}

  int Search(int n, const float* x, int topk, int* ids, float* distances);
  int Add(int n, float* x, int* ids);
  int Remove(int n, int* ids);
  int Reset();

 public:
  int d_;
  cnindexMetric_t metric_ = CNINDEX_METRIC_L2;
  int ntotal_;
  std::vector<float> xb_;
  std::vector<int> ids_;
};

int CpuFlat::Search(int n, const float* x, int topk, int* ids, float* distances) {
  typedef std::pair<int, float> Pair;
  std::vector<Pair> id_distance;

  for (size_t i = 0; i < (size_t)n; i++) {
    float L2_ids = 0.0;
    for (size_t j = 0; j < (size_t)ntotal_; j++) {
      float distance;
      if (metric_ == CNINDEX_METRIC_L2) {
        distance = cnindex::fvec_L2sqr(x + i * d_, xb_.data() + j * d_, d_);
      } else if (metric_ == CNINDEX_METRIC_IP) {
        distance = cnindex::fvec_inner_product(x + i * d_, xb_.data() + j * d_, d_);
      } else {
        std::cout << "CpuFlat unsupported metric type: " << metric_ << std::endl;
      }
      id_distance.push_back(std::make_pair(ids_[j], distance));
    }
    if (metric_ == CNINDEX_METRIC_L2) {
      std::partial_sort(id_distance.begin(), id_distance.begin() + std::min(topk, ntotal_), id_distance.end(),
                        [](Pair& x, Pair& y) -> bool { return x.second < y.second; });
    } else {
      std::partial_sort(id_distance.begin(), id_distance.begin() + std::min(topk, ntotal_), id_distance.end(),
                        [](Pair& x, Pair& y) -> bool { return x.second > y.second; });
    }
    for (int k = 0; k < topk; k++) {
      ids[i * topk + k] = k < ntotal_ ? id_distance[k].first : -1;
      distances[i * topk + k] = k < ntotal_ ? id_distance[k].second : std::numeric_limits<float>::max();
    }
    id_distance.clear();
  }
  return 0;
}

int CpuFlat::Add(int n, float* x, int* ids) {
  if (!ids) {
    std::cout << "[ERROR]CpuFlat Add need ids!" << std::endl;
    return -1;
  }
  xb_.insert(xb_.end(), x, x + (size_t)n * d_);
  ids_.insert(ids_.end(), ids, ids + n);
  ntotal_ += n;
  return 0;
}

int CpuFlat::Remove(int n, int* ids) {
  for (int i = 0; i < n; i++) {
    auto iter = std::find(ids_.begin(), ids_.end(), ids[i]);
    if (iter != ids_.end()) {
      size_t index = std::distance(ids_.begin(), iter);
      ids_[index] = ids_.back();
      ids_.erase(ids_.end() - 1);
      memcpy(xb_.data() + index * d_, xb_.data() + ((size_t)ntotal_ - 1) * d_, sizeof(float) * d_);
      xb_.erase(xb_.begin() + ((size_t)ntotal_ - 1) * d_, xb_.end());
      ntotal_--;
    } else {
      std::cout << "CpuFlat Remove() id: " << ids[i] << " is invalid\n";
      continue;
    }
  }
  return 0;
}

static int g_device_id = 0;

bool test_search(std::string mode_set, int nq, int d, int ntotal, int topk, int metric) {
  cnindexMetric_t metric_type = metric == 0 ? CNINDEX_METRIC_L2 : CNINDEX_METRIC_IP;
  // create cnindex flat
  cnindex::Flat mlu_flat(d, metric_type, g_device_id);

  // create cpu flat
  CpuFlat cpu_flat(d, metric_type);

  // prepare add vectors and ids
  std::vector<float> addvecs((size_t)ntotal * d);
  std::vector<int> ids(ntotal);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (size_t i = 0; i < (size_t)ntotal; i++) {
    for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
    ids[i] = i;
  }

#ifdef DUMP_DATA
  std::ofstream nlib_file("./nlib");
  char *nlib_data = (char *)addvecs.data();
  nlib_file.write(nlib_data, (size_t)ntotal * d * sizeof(float));
  nlib_file.close();
#endif

  // add
  cpu_flat.Add(ntotal, addvecs.data(), ids.data());
  auto ret = mlu_flat.Add(ntotal, addvecs.data(), ids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "MLU flat.Add() failed" << std::endl;
    return false;
  }

  {
    std::cout << "\n-------------- FLAT SEARCH ACCURACY TEST --------------" << std::endl;
    std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
              << "       nquery:   " << nq << std::endl
              << "       d:        " << d << std::endl
              << "       ntotal:   " << ntotal << std::endl
              << "       topk:     " << topk << std::endl << std::endl;

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

#ifdef DUMP_DATA
    std::ofstream nq_file("./nq");
    char *nq_data = (char *)query.data();
    nq_file.write(nq_data, (size_t)nq * d * sizeof(float));
    nq_file.close();
#endif
    // search accuracy
   cpu_flat.Search(nq, query.data(), topk, labels_cpu, distances_cpu);
#ifdef DUMP_DATA
    std::ofstream labels_file("./labels");
    char *labels_data = (char *)labels_cpu;
    labels_file.write(labels_data, (size_t)nq * topk * sizeof(int));
    labels_file.close();
    std::ofstream distances_file("./distances");
    char *distances_data = (char *)distances_cpu;
    distances_file.write(distances_data, (size_t)nq * topk * sizeof(float));
    distances_file.close();
#endif
   std::cout << "CPU Search OK" << std::endl;
   auto ret = mlu_flat.Search(nq, query.data(), topk, labels_mlu, distances_mlu);
   if (ret != CNINDEX_RET_SUCCESS) {
     std::cout << "mlu flat.Search() failed" << std::endl;
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
    std::cout << "\n------------ FLAT SEARCH PERFORMANCE TEST -------------" << std::endl;
    std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
              << "       nquery:   " << nq << std::endl
              << "       d:        " << d << std::endl
              << "       ntotal:   " << ntotal << std::endl
              << "       topk:     " << topk << std::endl << std::endl;

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

    // cpu search
    int cpu_iter_num = 10;
    double cpu_total_time = 0.0;
    for (int i = 0; i < cpu_iter_num; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      cpu_flat.Search(nq, query.data(), topk, labels_cpu, distances_cpu);
      std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
      cpu_total_time += elapsed.count();
    }
    std::cout << "CPU Search OK" << std::endl;

    int warm_iter_num = 1;
    int mlu_iter_num = 2;
    double mlu_total_time = 0.0;
    // mlu warm up
    for (int i = 0; i < warm_iter_num; i++) {
      int ret = mlu_flat.Search(1, query.data(), topk, labels_mlu, distances_mlu);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu flat.Search() failed" << std::endl;
        return false;
      }
    }
    // mlu search
    for (int i = 0; i < mlu_iter_num; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      int ret = mlu_flat.Search(nq, query.data(), topk, labels_mlu, distances_mlu);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu flat.Search() failed" << std::endl;
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

int test_add(int d, int add_num) {
  // create cnindex flat
  cnindex::Flat mlu_flat(d, CNINDEX_METRIC_L2, g_device_id);

  // create cpu flat
  CpuFlat cpu_flat(d, CNINDEX_METRIC_L2);

  {
    std::cout << "\n-------------- FLAT ADD ACCURACY TEST -----------------" << std::endl;
    std::cout << "ADD dataset:     " << "random"<< std::endl
              << "       add num:  " << add_num << std::endl
              << "       d:        " << d << std::endl << std::endl;

    // prepare add vectors and ids
    std::vector<float> addvecs((size_t)add_num * d);
    std::vector<int> ids(add_num);
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> c(-2.0, 2.0);
    for (size_t i = 0; i < (size_t)add_num; i++) {
      for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
      ids[i] = i;
    }

    // add
    cpu_flat.Add(add_num, addvecs.data(), ids.data());
    auto ret = mlu_flat.Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "MLU flat.Add() failed" << std::endl;
      return false;
    }

    int ntotal = mlu_flat.GetSize();
    if (ntotal != cpu_flat.ntotal_) {
      std::cout << "ADD failed: mlu ntotal != cpu ntotal" << std::endl;
      return false;
    }

    // compare codes ids
    std::vector<int> mlu_idx(ntotal);
    std::vector<float> mlu_vectors((size_t)ntotal * d);
    std::vector<int> cpu_idx;
    std::vector<float> cpu_vectors;
    mlu_flat.GetData(mlu_vectors.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_flat.ids_.data(), cpu_flat.ids_.data() + ntotal);
    cpu_vectors.insert(cpu_vectors.end(), cpu_flat.xb_.data(), cpu_flat.xb_.data() + (size_t)ntotal * d);

    std::pair<float, float> result = compare_vectors_ids(cpu_vectors.data(), cpu_idx.data(), mlu_vectors.data(), 
                                                         mlu_idx.data(), ntotal, d, true);
    std::cout << "all diff codes: " << (result.first > 0 ? RED : CYAN) << result.first << "%"
            << RESET << ", ids: " << (result.second > 0 ? RED : CYAN) << result.second << "%"
            << RESET << std::endl;
  }

  {
    std::cout << "\n-------------- FLAT ADD PERFORMANCE TEST --------------" << std::endl;
    std::cout << "Add dataset:    " << "ramdom" << std::endl
              << "    nadd:       " << add_num << std::endl
              << "    d:          " << d << std::endl << std::endl;

    mlu_flat.Reset();
    // prepare add vectors and ids
    std::vector<float> addvecs((size_t)add_num * d);
    std::vector<int> ids(add_num);
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> c(-2.0, 2.0);
    for (size_t i = 0; i < (size_t)add_num; i++) {
      for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
      ids[i] = i;
    }

    int warm_iter_num = 10;
    for (int i = 0; i < warm_iter_num; i++) {
      auto ret = mlu_flat.Add(1, addvecs.data(), ids.data());
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu flat.Add() failed" << std::endl;
        return false;
      }
    }
    mlu_flat.Reset();

    double add_total_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    auto ret = mlu_flat.Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu flat.Add() failed" << std::endl;
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

bool test_remove(int nremove, int d, int ntotal) {
  // create cnindex flat
  cnindex::Flat mlu_flat(d, CNINDEX_METRIC_L2, g_device_id);

  // create cpu flat
  CpuFlat cpu_flat(d, CNINDEX_METRIC_L2);

  // prepare add vectors and ids
  std::vector<float> addvecs((size_t)ntotal * d);
  std::vector<int> ids(ntotal);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (size_t i = 0; i < (size_t)ntotal; i++) {
    for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
    ids[i] = i;
  }

  // add
  cpu_flat.Add(ntotal, addvecs.data(), ids.data());
  auto ret = mlu_flat.Add(ntotal, addvecs.data(), ids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "MLU flat.Add() failed" << std::endl;
    return false;
  }

  {
    std::cout << "\n-------------- FLAT REMOVE ACCURACY TEST --------------" << std::endl;
    std::cout << "Remove dataset:  " << "random" << std::endl
              << "       nremove:  " << nremove << std::endl
              << "       d:        " << d << std::endl
              << "       ntotal:   " << ntotal << std::endl << std::endl;

    std::vector<int> remove_vecotr(nremove);
    for (int i = 0; i < nremove; i++) {
      remove_vecotr[i] = i;
    }

    int status = mlu_flat.Remove(nremove, remove_vecotr.data());
    int size_after_remove = mlu_flat.GetSize();
    cpu_flat.Remove(nremove, remove_vecotr.data());

    if (status != CNINDEX_RET_SUCCESS && size_after_remove != ntotal - nremove) {
      std::cout << "cnindex mlu FlatRemove failed" << std::endl;
    } else {
      std::cout << "cnindex FlatRemove sucess! ntotal = " << ntotal << ", remove = " << nremove 
                << ", after remove = " << size_after_remove << std::endl;
    }

    // compare codes ids
    int num_tmp = ntotal - nremove;
    std::vector<int> mlu_idx(num_tmp);
    std::vector<float> mlu_vectors(num_tmp * d);
    std::vector<int> cpu_idx;
    std::vector<float> cpu_vectors;
    mlu_flat.GetData(mlu_vectors.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_flat.ids_.data(), cpu_flat.ids_.data() + num_tmp);
    cpu_vectors.insert(cpu_vectors.end(), cpu_flat.xb_.data(), cpu_flat.xb_.data() + num_tmp * d);

    std::pair<float, float> result = compare_vectors_ids(cpu_vectors.data(), cpu_idx.data(), mlu_vectors.data(), 
                                                       mlu_idx.data(), num_tmp, d, true);
    std::cout << "all diff codes: " << (result.first > 0 ? RED : CYAN) << result.first * 100 << "%"
            << RESET << ", ids: " << (result.second > 0 ? RED : CYAN) << result.second * 100 << "%"
            << RESET << std::endl;
  }

  {
    std::cout << "\n-------------- FLAT REMOVE PERFORMANCE TEST --------------" << std::endl;
    std::cout << "Remove dataset: " << "ramdom" << std::endl
              << "    nremove:    " << nremove << std::endl
              << "    d:          " << d << std::endl << std::endl;

    mlu_flat.Reset();
    auto ret = mlu_flat.Add(ntotal, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu flat.Add() failed" << std::endl;
      return false;
    }

    // num for remove
    std::vector<int> remove_vecotr(nremove);
    for (int i = 0; i < nremove; i++) {
      remove_vecotr[i] = i;
    }

    double remove_total_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    ret = mlu_flat.Remove(nremove, remove_vecotr.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu flat.Remove() failed" << std::endl;
      return false;
    }
    std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
    remove_total_time += elapsed.count();

    std::cout << CYAN << "MLU Remove E2Etime: " << remove_total_time / nremove << "us, "
                  << nremove * 1E6 / remove_total_time << "qps" << RESET << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
  }

  return true;
}

int main(int argc, char* argv[]) {
  if (!(argc <= 8 && argc >= 6)) {
    printf("[ERROR] ./test_cnindex_flat device_id mode nq d ntotal topk metric(0: L2; 1: IP)\n");
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
  int ntotal = atoi(argv[5]);
  int topk = argc >= 7 ? atoi(argv[6]) : 1;
  int metric = argc >= 8 ? atoi(argv[7]) : 0;  // 0: L2; 1: IP
#else
  std::string mode_set = "r";
  int dataset = 1;
  int nq = 100000;
  int d = 256;
  int ntotal = 100000;
  int topk = 32;
  int metric = 0;
#endif

  if (mode_set == "a") {
    if (!test_add(d, ntotal)) return -1;
  } else if (mode_set == "r") {
    if (!test_remove(nq, d, ntotal)) return -1;
  } else {
    if (!test_search(mode_set, nq, d, ntotal, topk, metric)) return -1;
  }

  return 0;
}
