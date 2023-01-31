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

#include <sys/time.h>
#include "sys/sysinfo.h"
#include "sys/stat.h"

#include <cmath>
#include <cstddef>

#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include <utility>

#include "cnindex.h"
#include "../src/utils/arithmetics.h"
#include "../src/utils/distances.h"
#include "common.cpp"

class CpuIVFPQ {
 public:
  typedef std::pair<int32_t, float> Pair;
  CpuIVFPQ(int ntotal, int d, int nlist, int M, int nbits, int nprobe)
      : ntotal_(ntotal), d_(d), nlist_(nlist), M_(M), nbits_(nbits), nprobe_(nprobe) {
    nprobe_ = std::min(nprobe_, nlist_);
    code_size_ = (nbits_ * M + 7) / 8;
    dsub_ = d / M;
    ksub_ = 1 << nbits;
    nlist_size.resize(nlist);
  };
  ~CpuIVFPQ(){};
  void Search(int nq, const float *quary, int k, int *labels, float *distances);
  void Add(int n, const float *x, const int *labels);
  void Remove(int n, const int *remove_ids);

 public:
  std::vector<int> nlist_size;        // size<nlist>
  std::vector<uint8_t> codes;
  std::vector<int> ids;
  float *level1_centroids;
  std::vector<float> level2_centroids;

 public:
  int ntotal_ = 0;
  int d_ = 0;
  int nlist_ = 0;
  int M_ = 0;
  int nbits_ = 0;
  int nprobe_ = 0;
  int code_size_ = 0;
  int ksub_ = 0;
  int dsub_ = 0;
  int core_size_ = 0;
};

void CpuIVFPQ::Search(int nq, const float *quary, int k, int *labels, float *distances) {
  // when nlist == nprobe == 1, using core num search mode
  bool using_core_num = (nlist_ == 1) ? true : false;
  
  int nlist_loc = nlist_;
  int nprobe_loc = nprobe_;

  int core_num = 4;
  if (using_core_num) nlist_loc = nprobe_loc = core_num;

  // prepare data
  std::vector<std::vector<uint8_t>> all_codes;                     // level2
  std::vector<std::vector<int>> all_ids;
  std::unique_ptr<int[]> nprobe_ids(new int[nq * nprobe_loc]);            // level1
  std::unique_ptr<float[]> coarse_dis(new float[nq * nprobe_loc]);

  std::vector<float> residual(nq * nprobe_loc * d_);
  std::vector<Pair> pq_ids_distances;

  // if pq search mode, reset nlist_size
  if (using_core_num) {
    // copy query for nprobe_loc times of each nq
    for (int i = 0; i < nq; i++) {
      for (int j = 0; j < nprobe_loc; j++) {
        memcpy(residual.data() + i * nprobe_loc * d_ + j * d_, quary + i * d_, sizeof(float) * d_);
        nprobe_ids[i * nprobe_loc + j] = j;
      }
    }

    nlist_size.clear();
    int list_size = ntotal_ / nlist_loc;
    int list_end = ntotal_ % nlist_loc;
    for (int i = 0; i < nlist_loc; i++) {
      nlist_size.push_back(i < list_end ? (list_size + 1) : list_size);
    }
  } else {
    // every quary find topNprobe lists
    typedef std::pair<int32_t, float> Pair;
    std::vector<Pair> ids_distances;
    for (int q_i = 0; q_i < nq; q_i++) {
      // sort id with distances.
      float L2_dis = 0.0;
      for (int j = 0; j < nlist_; j++) {
        // every quary compary distances with all level1_centroids
        L2_dis = cnindex::fvec_L2sqr(quary + q_i * d_, level1_centroids + j * d_, d_);
        ids_distances.emplace_back(j, L2_dis);
      }
      partial_sort(ids_distances.begin(), ids_distances.begin() + nprobe_loc, ids_distances.end(),
                   [](Pair& x, Pair& y) -> bool { return x.second < y.second; });
      // copy topNprobe nprobe_ids.
      // [quary0-probe0, quary0-probe1,..., quary0-proben   // one quary have nprobe centroids
      //  ...
      //  quaryn-probe0, quaryn-probe1,..., quaryn-proben]
      for (int i = 0; i < nprobe_loc; i++) {
        nprobe_ids[i + q_i * nprobe_loc] = ids_distances[i].first;
      }

      // copy topNprobe coarse_dis.
      for (int i = 0; i < nprobe_loc; i++) {
        coarse_dis[i + q_i * nprobe_loc] = ids_distances[i].second;
      }

      // clear ids_distances
      ids_distances.clear();
    }
  }

  // prepare codes and ids offset
  std::vector<int> codes_offset, ids_offset;
  for (int i = 0; i < nlist_loc; i++) {
    codes_offset.push_back(i == 0 ? 0 : (codes_offset[i - 1] + nlist_size[i - 1] * code_size_));
    ids_offset.push_back(i == 0 ? 0 : (ids_offset[i - 1] + nlist_size[i - 1]));
  }

  // trans M * ksub * dsub -> ksub * d
  int ksub = 1 << nbits_;
  std::vector<float> level2_centroids_trans(ksub * d_);

  int dsub = d_ / M_;
  for (int i = 0; i < ksub; i++) {
    for (int j = 0; j < M_; j++) {
      int boundary = dsub * ksub * j + i * dsub;
      int dst_left = i * d_ + dsub * j;
      memcpy(level2_centroids_trans.data() + dst_left, level2_centroids.data() + boundary, dsub * sizeof(float));
    }
  }

  for (int i = 0; i < nq; i++) {
    std::vector<float> pq_distances;
    for (int j = 0; j < nprobe_loc; j++) {
      // compute residuals
      if (using_core_num) {
        cnindex::fvec_sub(quary + i * d_, level1_centroids, d_, residual.data() +  (i * nprobe_loc + j) * d_);
      } else {
        cnindex::fvec_sub(quary + i * d_, level1_centroids + nprobe_ids[i * nprobe_loc + j] * d_, d_,
            residual.data() +  (i * nprobe_loc + j) * d_);
      }
      // do pq search
      int nprobe_id = nprobe_ids[i * nprobe_loc + j];
      int list_size = nlist_size[nprobe_id];
      if (list_size <= 0) continue;
      pq_distances.resize(list_size);
      pq_search(residual.data() + i * d_ * nprobe_loc + j * d_, level2_centroids_trans.data(),
                codes.data() + codes_offset[nprobe_id], pq_distances.data(), 1, d_, k, list_size, code_size_);
      for (int idx = 0; idx < list_size; idx++) {
        pq_ids_distances.emplace_back(ids[ids_offset[nprobe_id] + idx], pq_distances[idx]);
      }
      pq_distances.clear();
    }

    int out_size = pq_ids_distances.size();
    int sort_end = std::min(out_size, k);
    partial_sort(pq_ids_distances.begin(), pq_ids_distances.begin() + sort_end, pq_ids_distances.end(),
                 [](Pair& x, Pair& y) -> bool { return x.second < y.second; });

    // output topk labels and distances
    for (int o = 0; o < k; o++) {
      labels[o + i * k] = o < out_size ? pq_ids_distances[o].first : -1;
      distances[o + i * k] = o < out_size ? pq_ids_distances[o].second : std::numeric_limits<float>::max();
    }

    pq_ids_distances.clear();
  }
}

void CpuIVFPQ::Add(int n, const float *x, const int *labels) {
  for (int x_idx = 0; x_idx < n; ++x_idx) {
    int list_idx = -1;
    float distance_min = 0;
    int same_distance_idx = -1;

    // choose list
    for (int i = 0; i < nlist_; i++) {
      float distance = cnindex::fvec_L2sqr(x + x_idx * d_, level1_centroids + i * d_, d_);
      if (distance_min == distance) same_distance_idx = i;
      if (i == 0 || distance < distance_min) {
        list_idx = i;
        distance_min = distance;
        same_distance_idx = -1;
      }
    }
    if (same_distance_idx >= 0) {
      std::cout << "CpuIVFPQ::Add() [Warning] x[" << x_idx << "] id=" << labels[x_idx]
                << " has same distances between level1 centroid " << list_idx << " and " << same_distance_idx
                << std::endl;
    }

    // compute residual
    std::vector<float> residual(d_);
    cnindex::fvec_sub(x + x_idx * d_, level1_centroids + list_idx * d_, d_, residual.data());

    // pq encode
    std::vector<uint8_t> code;
    for (int m_idx = 0; m_idx < M_; m_idx++) {
      distance_min = 0;
      uint8_t idx_min = 0;
      for (int k_idx = 0; k_idx < ksub_; k_idx++) {
        float *level2_centroid = level2_centroids.data() + (m_idx * ksub_ + k_idx) * dsub_;
        float distance = cnindex::fvec_L2sqr(residual.data() + m_idx * dsub_, level2_centroid, dsub_);
        if (k_idx == 0 || distance < distance_min) {
          distance_min = distance;
          idx_min = k_idx;
        }
      }
      code.push_back(idx_min);
    }

    // insert encoded vector to the tail of list
    int offset = std::accumulate(nlist_size.begin(), nlist_size.begin() + list_idx + 1, 0);
    codes.insert(codes.begin() + (size_t)offset * M_, code.begin(), code.end());
    ids.insert(ids.begin() + offset, labels[x_idx]);
    nlist_size[list_idx]++;
  }
}

void CpuIVFPQ::Remove(int n, const int *remove_ids) {
  for (int i = 0; i < n; i++) {
    int id = remove_ids[i];
    for (int nlist_idx = 0; nlist_idx < nlist_; nlist_idx++) {
      int list_size = nlist_size[nlist_idx];
      int offset = std::accumulate(std::begin(nlist_size), std::begin(nlist_size) + nlist_idx, 0);
      std::vector<int> id_vec(std::begin(ids) + offset, std::begin(ids) + offset + list_size);
      auto iter = std::find(std::begin(id_vec), std::end(id_vec), id);
      if(iter == std::end(id_vec)) {
        continue;
      }else{
        int idx = std::distance(std::begin(id_vec), iter);
        for (int m_idx = 0; m_idx < M_; m_idx++) {
          codes[(size_t)(offset + idx) * M_ + m_idx] = codes[(size_t)(offset + list_size - 1) * M_ + m_idx];
        }
        codes.erase(std::begin(codes) + (size_t)(offset + list_size - 1) * M_,
                    std::begin(codes) + (size_t)(offset + list_size) * M_);
        ids[offset + idx] = ids[offset + list_size - 1];
        ids.erase(ids.begin() + offset + list_size - 1);
        nlist_size[nlist_idx] -= 1;
        ntotal_--;
        break;
      }
    }
  }
}

typedef struct {
  std::vector<float> query;
  std::vector<float>level1_centroids;
  std::vector<float> level2_centroids;
  std::vector<int> nlist_size; 
  std::vector<uint8_t> codes;
  std::vector<int> ids;
} Data;

void prepare_random_data(Data &data, int d, int M, int nbits, int ntotal, int nlist) {
  int code_size = (nbits * M + 7) / 8;
  int ksub = 1 << nbits;

  data.level1_centroids.resize(nlist * d);
  data.level2_centroids.resize(d * ksub);
  data.nlist_size.resize(nlist); 
  data.codes.resize((size_t)ntotal * code_size);
  data.ids.resize(ntotal);

  srand(35);
  int list_size = ntotal / nlist;
  int lr;
  for (int i = 0; i < nlist; i++) {
    data.nlist_size[i] = i < (ntotal % nlist) ? (list_size + 1) : list_size;
#if 0
    if (nlist > 1 && !(nlist % 2 && i == (nlist - 1))) {
      if (i % 2 == 0) {
        if (list_size <= 1) {
          lr = 0;
        } else {
          lr = rand() % (list_size - 1);
        }
        data.nlist_size[i] += lr;
      } else {
        data.nlist_size[i] -= lr;
      }
    }
#endif
  }
#if 0
  // make list_size[0] = 0 to test empty list case
  data.nlist_size[1] += data.nlist_size[0];
  data.nlist_size[0] = 0;
#endif
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c1(-1.0, 1.0);
  std::uniform_real_distribution<float> c2(-2.0, 2.0);
  for (int i = 0; i < nlist * d; i++) {
    data.level1_centroids[i] = c1(e);
  }
  for (int i = 0; i < d * ksub; i++) {
    data.level2_centroids[i] = c2(e);
  }
  for (size_t i = 0; i < (size_t)ntotal * code_size; i++) {
    data.codes[i] = rand() % 256;
  }
  for (int i = 0; i < ntotal; i++) {
    data.ids[i] = i;
  }
}

void prepare_real_data(Data &data, const std::string &dataset, int d, int M, int nbits, int ntotal, int nlist) {
  int code_size = (nbits * M + 7) / 8;
  int ksub = 1 << nbits;

  data.level1_centroids.resize(nlist * d);
  data.level2_centroids.resize(d * ksub);
  data.nlist_size.resize(nlist); 
  data.codes.resize((size_t)ntotal * code_size);
  data.ids.resize(ntotal);

  read_data(dataset + "/nlist_size", data.nlist_size.data(), nlist);
  read_data(dataset + "/level1_centroids", data.level1_centroids.data(), nlist * d);
  read_data(dataset + "/level2_centroids", data.level2_centroids.data(), d * ksub);
  read_lib(dataset + "/codes", data.codes.data(), ntotal * code_size);
  read_data(dataset + "/ids", data.ids.data(), ntotal);
}

void prepare_query(Data &data, const std::string &dataset, int nq, int d) {
  int nquery = nq * d;
  data.query.resize(nquery);

  if (dataset == "s") {  // random data
    std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> q(-3.0, 3.0);
    for (int i = 0; i < nq * d; i++) {
      data.query[i] = q(e);
    }
  } else {
    read_data(dataset + "/query", data.query.data(), nquery);
  }
}

bool compare_lists_codes_ids(CpuIVFPQ &cpu_ivfpq, cnindex::IVFPQ &ivfpq) {
  int nlist = cpu_ivfpq.nlist_;
  int m = cpu_ivfpq.M_;
  int nq = cpu_ivfpq.ntotal_;

  // get codes ids
  std::vector<int> mlu_list_size(nlist);
  std::vector<int> cpu_list_size(nlist);
  for (int i = 0; i < nlist; i++) {
    mlu_list_size[i] = ivfpq.GetListSize(i);
    cpu_list_size[i] = cpu_ivfpq.nlist_size[i];
  }

  // compare list size
  float list_size_diff = compare(cpu_list_size.data(), mlu_list_size.data(), nlist, 1);
  std::cout << CYAN << "List size diff: " << (list_size_diff > 0 ? RED : CYAN) << list_size_diff * 100 << "%"
            << RESET << std::endl;
  if (list_size_diff != 0) {
    std::cout << "CPU list size: " << std::endl;
    for (int i = 0; i < nlist; i++) {
      if (cpu_list_size[i] != mlu_list_size[i]) {
        std::cout << CYAN << cpu_list_size[i] << " " << RESET;
      } else {
        std::cout << cpu_list_size[i] << " ";
      }
    }
    std::cout << std::endl;
    std::cout << "MLU list size: " << std::endl;
    for (int i = 0; i < nlist; i++) {
      if (cpu_list_size[i] != mlu_list_size[i]) {
        std::cout << RED << mlu_list_size[i] << " " << RESET;
      } else {
        std::cout << mlu_list_size[i] << " ";
      }
    }
    std::cout << std::endl;
    return false;
  }

  std::vector<int> mlu_idx;
  std::vector<uint8_t> mlu_codes;
  std::vector<int> cpu_idx;
  std::vector<uint8_t> cpu_codes;
  std::vector<int> list_offset(nlist);
  for (int i = 0; i < nlist; i++) {
    list_offset[i] = (i == 0) ? 0 : (list_offset[i - 1] + mlu_list_size[i - 1]);
    std::vector<int> ids(mlu_list_size[i]);
    std::vector<uint8_t> codes((size_t)mlu_list_size[i] * m);
    ivfpq.GetListData(i, codes.data(), ids.data());
    mlu_idx.insert(mlu_idx.end(), ids.data(), ids.data() + mlu_list_size[i]);
    mlu_codes.insert(mlu_codes.end(), codes.data(), codes.data() + (size_t)mlu_list_size[i] * m);
    ids.clear();
    codes.clear();
  }
  cpu_idx.insert(cpu_idx.end(), cpu_ivfpq.ids.data(), cpu_ivfpq.ids.data() + nq);
  cpu_codes.insert(cpu_codes.end(), cpu_ivfpq.codes.data(), cpu_ivfpq.codes.data() + nq * m);

  int code_diff = 0, id_diff = 0;
  for (int i = 0; i < nlist; i++) {
    if (mlu_list_size[i] == 0)
      continue;
    std::pair<float, float> result = compare_codes_ids(
      cpu_codes.data() + (size_t)list_offset[i] * m, cpu_idx.data() + list_offset[i],
      mlu_codes.data() + (size_t)list_offset[i] * m, mlu_idx.data() + list_offset[i],
      (size_t)mlu_list_size[i], m, true);

    if (result.first > 0 || result.second > 0) {
      std::cout << RED << "list[" << i << "] has diff" << RESET << std::endl;
      code_diff += result.first * mlu_list_size[i];
      id_diff += result.second * mlu_list_size[i];
    }
    if (result.first == -1 || result.second == -1) return false;
  }

  double code_diff_precision = code_diff == 0 ? 0 : (code_diff * 100.0 / nq);
  double id_diff_precision = id_diff == 0 ? 0 : (id_diff * 100.0 / nq);

  std::cout << "all diff codes: " << (code_diff > 0 ? RED : CYAN) << code_diff_precision << "%"
            << RESET << ", ids: " << (id_diff > 0 ? RED : CYAN) << id_diff_precision << "%"
            << RESET << std::endl;

  if (code_diff_precision > 0.001) return false;

  return true;
}

static int g_device_id = 0;

bool test_search(const std::string &mode_set, int nq, int d, int M, int nbits,
                      int ntotal, int nlist, int nprobe, int topk) {
  if (nq <= 0) {
    std::cout << "[ERROR] nq <= 0!";
    return false;
  }

  Data data;
  if (mode_set == "s") {
    prepare_random_data(data, d, M, nbits, ntotal, nlist);
  } else {
    prepare_real_data(data, mode_set, d, M, nbits, ntotal, nlist);
  }

  // CPU IVFPQ
  CpuIVFPQ cpu_ivfpq(ntotal, d, nlist, M, nbits, nprobe);
  cpu_ivfpq.nlist_size = data.nlist_size;
  cpu_ivfpq.level1_centroids = data.level1_centroids.data();
  cpu_ivfpq.level2_centroids = data.level2_centroids;
  cpu_ivfpq.codes = data.codes;
  cpu_ivfpq.ids = data.ids;

  // MLU IVFPQ
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, g_device_id);
  int ret = flat.Add(nlist, data.level1_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "flat.Add() failed" << std::endl;
    return false;
  }
  cnindex::IVFPQ ivfpq(&flat, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  ret = ivfpq.SetCentroids(data.level2_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.SetCentroids() failed" << std::endl;
    return false;
  }

  int offset = 0;
  for (int i = 0; i < data.nlist_size.size(); i++) {
    ivfpq.SetListData(i, data.nlist_size[i], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
    offset += data.nlist_size[i];
  }

  // search result
  int *labels_cpu = new int[(size_t)nq * topk];
  std::unique_ptr<int> lcup(labels_cpu);
  float *distances_cpu = new float[(size_t)nq * topk];
  std::unique_ptr<float> dcup(distances_cpu);
  int *labels_mlu = new int[(size_t)nq * topk];
  std::unique_ptr<int> lmup(labels_mlu);
  float* distances_mlu = new float[(size_t)nq * topk];
  std::unique_ptr<float> dmup(distances_mlu);

  prepare_query(data, mode_set, nq, d);

  std::cout << "\n-------------- IVFPQ SEARCH ACCURACY TEST --------------" << std::endl;
  std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
            << "       nquery:   " << nq << std::endl
            << "       d:        " << d << std::endl
            << "       M:        " << M << std::endl
            << "       nbits:    " << nbits << std::endl
            << "       ntotal:   " << ntotal << std::endl
            << "       nlist:    " << nlist << std::endl
            << "       nprobe:   " << nprobe << std::endl
            << "       topk:     " << topk << std::endl << std::endl;

  // search accuracy
  cpu_ivfpq.Search(nq, data.query.data(), topk, labels_cpu, distances_cpu);
  std::cout << "CPU Search OK" << std::endl;
  ret = ivfpq.Search(nq, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Search() failed" << std::endl;
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

  std::cout << "\n------------ IVFPQ SEARCH PERFORMANCE TEST -------------" << std::endl;
  std::cout << "Search dataset:  " << (mode_set == "s" ? "random" : mode_set) << std::endl
            << "       nquery:   " << nq << std::endl
            << "       d:        " << d << std::endl
            << "       M:        " << M << std::endl
            << "       nbits:    " << nbits << std::endl
            << "       ntotal:   " << ntotal << std::endl
            << "       nlist:    " << nlist << std::endl
            << "       nprobe:   " << nprobe << std::endl
            << "       topk:     " << topk << std::endl << std::endl;

  // cpu search
  int cpu_iter_num = 2;
  double cpu_total_time = 0.0;
  for (int i = 0; i < cpu_iter_num; i++) {
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    cpu_ivfpq.Search(nq, data.query.data(), topk, labels_cpu, distances_cpu);
    std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
    cpu_total_time += elapsed.count();
  }
  std::cout << "CPU Search OK" << std::endl;

  int warm_iter_num = 1;
  int mlu_iter_num = 2;
  double mlu_total_time = 0.0;
  // mlu warm up
  for (int i = 0; i < warm_iter_num; i++) {
    int ret = ivfpq.Search(1, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Search() failed" << std::endl;
      return false;
    }
  }
  // mlu search
  for (int i = 0; i < mlu_iter_num; i++) {
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int ret = ivfpq.Search(nq, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Search() failed" << std::endl;
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

  return true;
}

bool test_add(int nq, int d, int M, int nbits, int ntotal, int nlist) {
  Data data;
  prepare_random_data(data, d, M, nbits, 0, nlist);

  std::cout << "\n---------------- IVFPQ ADD ACCURACY TEST ---------------" << std::endl;
  std::cout << "Add dataset:  " << "ramdom" << std::endl
            << "    nadd:     " << ntotal << std::endl
            << "    d:        " << d << std::endl
            << "    M:        " << M << std::endl
            << "    nbits:    " << nbits << std::endl
            << "    ntotal:   " << ntotal << std::endl
            << "    nlist:    " << nlist << std::endl << std::endl;

  // CPU IVFPQ
  CpuIVFPQ cpu_ivfpq(ntotal, d, nlist, M, nbits, 128);
  cpu_ivfpq.level1_centroids = data.level1_centroids.data();
  cpu_ivfpq.level2_centroids = data.level2_centroids;

  // MLU IVFPQ
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, g_device_id);
  int ret = flat.Add(nlist, data.level1_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "flat.Add() failed" << std::endl;
    return false;
  }
  cnindex::IVFPQ ivfpq(&flat, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  ret = ivfpq.SetCentroids(data.level2_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.SetCentroids() failed" << std::endl;
    return false;
  }

  // prepare add vectors and ids
  std::vector<float> addvecs((size_t)ntotal * d);
  std::vector<int> ids(ntotal);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> q(-3.0, 3.0);
  for (size_t i = 0; i < (size_t)ntotal; i++) {
    for (int j = 0; j < d; j++) addvecs[i * d + j] = q(e);
    ids[i] = i;
  }

  // add
  cpu_ivfpq.Add(ntotal, addvecs.data(), ids.data());
  std::cout << "CPU Add OK" << std::endl;
  ret = ivfpq.Add(ntotal, addvecs.data(), ids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Add() failed" << std::endl;
    return false;
  }
  std::cout << "MLU Add OK" << std::endl;
  if (!compare_lists_codes_ids(cpu_ivfpq, ivfpq)) return false;

  std::cout << "\n-------------- IVFPQ ADD PERFORMANCE TEST --------------" << std::endl;
  std::cout << "Add dataset:    " << "ramdom" << std::endl
            << "    nadd:       " << ntotal << std::endl
            << "    d:          " << d << std::endl
            << "    M:          " << M << std::endl
            << "    nbits:      " << nbits << std::endl
            << "    ntotal:     " << ntotal << std::endl
            << "    nlist:      " << nlist << std::endl << std::endl;

  int warm_iter_num = 1;
  for (int i = 0; i < warm_iter_num; i++) {
    ret = ivfpq.Add(1, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Add() failed" << std::endl;
      return false;
    }
  }
  ret = ivfpq.Reset();
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Reset() failed" << std::endl;
    return false;
  }
  double add_total_time = 0.0;
  std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
  ret = ivfpq.Add(ntotal, addvecs.data(), ids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Add() failed" << std::endl;
    return false;
  }
  std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
  add_total_time += elapsed.count();

  std::cout << CYAN << "MLU Add E2Etime: " << add_total_time / ntotal << "us, "
                << ntotal * 1E6 / add_total_time << "qps" << RESET << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;
  
  return true;
}

bool test_remove(int nremove, int d, int M, int nbits, int ntotal, int nlist) {
  Data data;
  prepare_random_data(data, d, M, nbits, ntotal, nlist);

  std::cout << "\n-------------- IVFPQ REMOVE ACCURACY TEST --------------" << std::endl;
  std::cout << "Remove dataset:    " << "ramdom" << std::endl
            << "       nremove:    " << nremove << std::endl
            << "       d:          " << d << std::endl
            << "       M:          " << M << std::endl
            << "       nbits:      " << nbits << std::endl
            << "       ntotal:     " << ntotal << std::endl
            << "       nlist:      " << nlist << std::endl << std::endl;

  // CPU IVFPQ
  CpuIVFPQ cpu_ivfpq(ntotal, d, nlist, M, nbits, 128);
  cpu_ivfpq.level1_centroids = data.level1_centroids.data();
  cpu_ivfpq.level2_centroids = data.level2_centroids;
  cpu_ivfpq.nlist_size = data.nlist_size;
  cpu_ivfpq.ids = data.ids;
  cpu_ivfpq.codes = data.codes;

  // MLU IVFPQ
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, g_device_id);
  int ret = flat.Add(nlist, data.level1_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "flat.Add() failed" << std::endl;
    return false;
  }
  cnindex::IVFPQ ivfpq(&flat, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  ret = ivfpq.SetCentroids(data.level2_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.SetCentroids() failed" << std::endl;
    return false;
  }

  int offset = 0;
  for (int i = 0; i < data.nlist_size.size(); i++) {
    ivfpq.SetListData(i, data.nlist_size[i], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
    offset += data.nlist_size[i];
  }

  // genrate unduplicated remove ids from total ids
  srand(35);
  std::vector<int> total_ids(data.ids);
  std::vector<int> remove_ids;
  for (int i = 0; i < nremove; i++) {
    int idx = total_ids.size() > 1 ? rand() % (total_ids.size() - 1) : 0;
    remove_ids.push_back(total_ids[idx]);
    total_ids[idx] = total_ids.back();
    total_ids.erase(total_ids.end() - 1);
  }

  // Remove
  cpu_ivfpq.Remove(nremove, remove_ids.data());
  std::cout << "CPU Remove OK" << std::endl;
  ret = ivfpq.Remove(nremove, remove_ids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Remove() failed" << std::endl;
    return false;
  }
  std::cout << "MLU Remove OK" << std::endl;
  if (!compare_lists_codes_ids(cpu_ivfpq, ivfpq)) return false;

  // the list_size after remove
  int size_after_remove = 0;
  for (int i = 0; i < nlist; i++) {
    size_after_remove += ivfpq.GetListSize(i);
  }
  std::cout << "MLU REMOVE RESULT : "
            << " add num = " << ntotal << "  remove num = " << nremove
            << "  add - remove = " << size_after_remove << std::endl;

  std::cout << "\n------------ IVFPQ REMOVE PERFORMANCE TEST -------------" << std::endl;
  std::cout << "Remove dataset:    " << "ramdom" << std::endl
            << "       nremove:    " << nremove << std::endl
            << "       d:          " << d << std::endl
            << "       M:          " << M << std::endl
            << "       nbits:      " << nbits << std::endl
            << "       ntotal:     " << ntotal << std::endl
            << "       nlist:      " << nlist << std::endl << std::endl;
  // warm up
  int warm_iter_num = 10;
  ret = ivfpq.Reset();
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Reset() failed" << std::endl;
    return false;
  }
  offset = 0;
  for (int i = 0; i < data.nlist_size.size(); i++) {
    ivfpq.SetListData(i, data.nlist_size[i], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
    offset += data.nlist_size[i];
  }
  for (int i = 0; i < warm_iter_num; i++) {
    ivfpq.Remove(1, remove_ids.data() + i);
  }

  // mlu remove
  double remove_total_time = 0.0;
  ret = ivfpq.Reset();
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.Reset() failed" << std::endl;
    return false;
  }
  offset = 0;
  for (int i = 0; i < data.nlist_size.size(); i++) {
    ivfpq.SetListData(i, data.nlist_size[i], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
    offset += data.nlist_size[i];
  }
  double test_min_time = std::numeric_limits<double>::max(), test_max_time = 0;
  for (int i = 0; i < nremove; i++) {
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    ret = ivfpq.Remove(1, remove_ids.data() + i);
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Remove() failed" << std::endl;
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

  return true;
}

bool test_multi_instances(int nq, int d, int M, int nbits, int ntotal, int nlist, int nprobe, int topk) {
  if (nq <= 0) {
    std::cout << "[ERROR] nq <= 0!";
    return false;
  }

  int instances_num = 100;
  int warm_iter_num = 10;

  double qps_min = std::numeric_limits<double>::max();
  double qps_max = 0;
  double qps_total = 0;

  Data data;
  prepare_random_data(data, d, M, nbits, ntotal, nlist);

  // prepare add vectors and ids
  std::vector<float> addvecs(ntotal * d);
  std::vector<int> ids(ntotal);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> q(-3.0, 3.0);
  for (int i = 0; i < ntotal; i++) {
    for (int j = 0; j < d; j++) addvecs[i * d + j] = q(e);
    ids[i] = i;
  }

  // create
  std::vector<cnindex::Flat> flats;
  std::vector<cnindex::IVFPQ> ivfpqs;
  int ret;
  std::cout << "creating MLU IVFPQ index" << std::endl;
  flats.reserve(instances_num);
  ivfpqs.reserve(instances_num);
  for (int i = 0; i < instances_num; i++) {
    std::cout << "[" << i << "]";
    flats.emplace_back(d, CNINDEX_METRIC_L2, g_device_id);
    auto &flat = flats.back();
    ret = flat.Add(nlist, data.level1_centroids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "flat.Add() failed" << std::endl;
      return false;
    }

    ivfpqs.emplace_back(&flat, CNINDEX_METRIC_L2, M, nbits, g_device_id);
    auto &ivfpq = ivfpqs.back();
    ret = ivfpq.SetCentroids(data.level2_centroids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.SetCentroids() failed" << std::endl;
      return false;
    }
#if 1
    if (i == 0) {  // mlu warm up
      std::cout << "\n------------ IVFPQ MULTI INSTANCES ADD TEST -----------" << std::endl;
      std::cout << "       nadd:     " << ids.size() << std::endl
                << "       d:        " << d << std::endl
                << "       M:        " << M << std::endl
                << "       nbits:    " << nbits << std::endl
                << "       ntotal:   " << ntotal << std::endl
                << "       nlist:    " << nlist << std::endl
                << "       nprobe:   " << nprobe << std::endl
                << "       topk:     " << topk << std::endl << std::endl;

      for (int i = 0; i < warm_iter_num; i++) {
        ret = ivfpq.Add(1, addvecs.data() + i * d, ids.data() + i);
        if (ret != CNINDEX_RET_SUCCESS) {
          std::cout << "ivfpq.Add() failed" << std::endl;
          return false;
        }
      }
      ret = ivfpq.Reset();
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "ivfpq.Reset() failed" << std::endl;
        return false;
      }
    }

    // mlu add
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    ret = ivfpq.Add(ids.size(), addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Add() failed" << std::endl;
      return false;
    }
    std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
    double add_time = elapsed.count();

    double qps = ids.size() * 1E6 / add_time;
    qps_min = std::min(qps_min, qps);
    qps_max = std::max(qps_max, qps);
    qps_total += qps;

    std::cout << "MLU Add [" << i << "] OK" << std::endl;
    std::cout << CYAN << "E2Etime: " << add_time << "us, " << qps << "qps" << RESET << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
#else
    qps_min = 0;
    int offset = 0;
    for (int j = 0; j < data.nlist_size.size(); j++) {
      ivfpq.SetListData(j, data.nlist_size[j], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
      offset += data.nlist_size[j];
    }
#endif
  }

  std::cout << "MLU IVFPQ indices created" << std::endl;

  std::cout << "-------------------------------------------------------" << std::endl;
  std::cout << CYAN << "MLU Add throughput: min=" << qps_min << "qps, max=" << qps_max << "qps, avg="
            << (qps_total / ivfpqs.size()) << "qps" << RESET << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;

  // search
  int *labels_mlu = new int[(size_t)nq * topk];
  std::unique_ptr<int> lmup(labels_mlu);
  float* distances_mlu = new float[(size_t)nq * topk];
  std::unique_ptr<float> dmup(distances_mlu);

  qps_min = std::numeric_limits<double>::max();
  qps_max = 0;
  qps_total = 0;

  prepare_query(data, "s", nq, d);

  std::cout << "\n------------ IVFPQ MULTI INSTANCES SEARCH TEST -----------" << std::endl;
  std::cout << "       nquery:   " << nq << std::endl
            << "       d:        " << d << std::endl
            << "       M:        " << M << std::endl
            << "       nbits:    " << nbits << std::endl
            << "       ntotal:   " << ntotal << std::endl
            << "       nlist:    " << nlist << std::endl
            << "       nprobe:   " << nprobe << std::endl
            << "       topk:     " << topk << std::endl << std::endl;

  // search
  for (auto &ivfpq : ivfpqs) {
    int idx = std::distance(ivfpqs.data(), &ivfpq);
    // mlu warm up
    for (int i = 0; i < warm_iter_num; i++) {
      ret = ivfpq.Search(1, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "ivfpq.Search() failed" << std::endl;
        return false;
      }
    }
    // mlu search
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    ret = ivfpq.Search(nq, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Search() failed" << std::endl;
      return false;
    }
    std::chrono::duration<double, std::micro> elapsed = std::chrono::system_clock::now() - start;
    double search_time = elapsed.count();

    double qps = nq * 1E6 / search_time;
    qps_min = std::min(qps_min, qps);
    qps_max = std::max(qps_max, qps);
    qps_total += qps;

    std::cout << "MLU Search [" << idx << "] OK" << std::endl;
    std::cout << CYAN << "E2Etime: " << search_time << "us, " << qps << "qps" << RESET << std::endl;
  }

  std::cout << "-------------------------------------------------------" << std::endl;
  std::cout << CYAN << "MLU Search throughput: min=" << qps_min << "qps, max=" << qps_max << "qps, avg="
            << (qps_total / ivfpqs.size()) << "qps" << RESET << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;

  ivfpqs.clear();
  flats.clear();
}

void test_concurrency() {
  int nq = 32;
  int d = 256;
  int M = 32;
  int nbits = 8;
  int ntotal = 1000000;
  int nlist = 1024;
  int nprobe = 128;
  int topk = 32;
  int ret;
  Data data;
  prepare_random_data(data, d, M, nbits, ntotal, nlist);
  prepare_query(data, "s", nq, d);

  std::cout << "\n---------------- IVFPQ CONCURRENCY TEST ------------------" << std::endl;

  auto search_task = [] (cnindex::IVFPQ *ivfpq, int nq, Data &data, int nprobe, int topk, int &ret) {
    int *labels_mlu = new int[(size_t)nq * topk];
    std::unique_ptr<int> lmup(labels_mlu);
    float* distances_mlu = new float[(size_t)nq * topk];
    std::unique_ptr<float> dmup(distances_mlu);
    std::cout << "start search task" << std::endl;
    ret = ivfpq->Search(nq, data.query.data(), nprobe, topk, labels_mlu, distances_mlu);
    std::cout << "end search task" << std::endl;
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "ivfpq.Search() failed" << std::endl;
    }
  };

  std::cout << "start create" << std::endl;
  cnindex::Flat flat(d, CNINDEX_METRIC_L2, g_device_id);
  ret = flat.Add(nlist, data.level1_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "flat.Add() failed" << std::endl;
    return;
  }
  cnindex::IVFPQ *ivfpq = new cnindex::IVFPQ(&flat, CNINDEX_METRIC_L2, M, nbits, g_device_id);
  ret = ivfpq->SetCentroids(data.level2_centroids.data());
  if (ret != CNINDEX_RET_SUCCESS) {
    std::cout << "ivfpq.SetCentroids() failed" << std::endl;
    delete ivfpq;
    return;
  }
  std::cout << "end create" << std::endl;

  std::cout << "start set data" << std::endl;
  int offset = 0;
  for (int i = 0; i < data.nlist_size.size(); i++) {
    ivfpq->SetListData(i, data.nlist_size[i], data.codes.data() + (size_t)offset * M, data.ids.data() + offset);
    offset += data.nlist_size[i];
  }
  std::cout << "end set data" << std::endl;

  std::thread t = std::thread(search_task, ivfpq, nq, std::ref(data), nprobe, topk, std::ref(ret));
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::cout << "start destroy" << std::endl;
  delete ivfpq;
  std::cout << "end destroy" << std::endl;

  if (t.joinable()) t.join();
  // break search will return CNINDEX_FAILED
  std::cout << CYAN << "Concurrency test result: " << (ret != CNINDEX_RET_SUCCESS ? "Passed" : (RED"Failed"))
            << RESET << std::endl;
}

int main(int argc, char* argv[]) {
  if (!((argc <= 11 && argc >= 10) || (argc >= 3 && std::string(argv[2]) == "c"))) {
    printf("[ERROR] ./test_cnindex_ivfpq device_id mode nq d M nbits ntotal nlist nprobe [topk/nbatch]\n");
    printf("         mode: s: search test using random dataset\n"
           "               \"test-data\": search test using dataset from files\n"
           "               a: add test, last parameter is nbatch\n"
           "               r: remove test\n"
           "               m: multi instances search test\n"
           "               c: concurrency test\n");
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
  int nlist = atoi(argv[8]);
  int nprobe = atoi(argv[9]);
  int topk = argc == 11 ? atoi(argv[10]) : 1;
#else
  std::string mode_set = "r";
  int dataset = 1;
  int nq = 100000;
  int d = 256;
  int M = 32;
  int nbits = 8;
  int ntotal = 100000;
  int nlist = 1024;
  int nprobe = 128;
  int topk = 32;
#endif

  if (mode_set == "c") {
    test_concurrency();
  } else if (mode_set == "m") {
    test_multi_instances(nq, d, M, nbits, ntotal, nlist, nprobe, topk);
  } else if (mode_set == "a") {
    if (!test_add(ntotal, d, M, nbits, ntotal, nlist)) return -1;
  } else if (mode_set == "r") {
    if (!test_remove(nq, d, M, nbits, ntotal, nlist)) return -1;
  } else {
    if (!test_search(mode_set, nq, d, M, nbits, ntotal, nlist, nprobe, topk)) return -1;
  }

  return 0;
}
