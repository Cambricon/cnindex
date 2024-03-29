#ifndef CPU_PQ_FLAT_CPP
#define CPU_PQ_FLAT_CPP
#include <fstream>
#include <random>
#include <chrono>
#include "cpu_pq_flat.hpp"
#include "../src/utils/distances.h"
#include "cnindex.h"
#include "../tests/common.cpp"

int CpuPQ::Add(int n, const float *x, const int *ids)
{
  if (n <= 0 || !x || !ids)
  {
    std::cout << "[ERROR]CpuPQ::Add invalid parameters!" << std::endl;
    return -1;
  }

  for (int i = 0; i < n; i++)
  {
    float distance_min = 0;
    // pq encode
    std::vector<uint8_t> code;
    for (int j = 0; j < M_; j++)
    {
      distance_min = 0;
      uint8_t idx_min = 0;
      for (int k = 0; k < ksub_; k++)
      {
        float *centroids_sub_k = centroids_.data() + (j * ksub_ + k) * dsub_;
        float distance = cnindex::fvec_L2sqr(x + i * d_ + j * dsub_, centroids_sub_k, dsub_);
        if (k == 0 || distance < distance_min)
        {
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

int CpuPQ::Search(int n, float *x, int topk, int *labels, float *distances)
{
  // trans M * ksub * dsub -> ksub * d
  std::vector<float> centroids_trans(ksub_ * d_);
  typedef std::pair<int32_t, float> Pair;
  std::vector<Pair> pq_ids_distances;

  for (int i = 0; i < ksub_; i++)
  {
    for (int j = 0; j < M_; j++)
    {
      int boundary = dsub_ * ksub_ * j + i * dsub_;
      int dst_left = i * d_ + dsub_ * j;
      memcpy(centroids_trans.data() + dst_left, centroids_.data() + boundary, dsub_ * sizeof(float));
    }
  }

  for (int i = 0; i < n; i++)
  {
    std::vector<float> pq_distances;

    // do pq search
    pq_distances.resize(ntotal_);
    pq_search(x + i * d_, centroids_trans.data(), codes_.data(), pq_distances.data(), 1, d_, topk,
              ntotal_, code_size_);
    for (int idx = 0; idx < ntotal_; idx++)
    {
      pq_ids_distances.emplace_back(ids_[idx], pq_distances[idx]);
    }
    pq_distances.clear();

    int out_size = pq_ids_distances.size();
    int sort_end = std::min(out_size, topk);
    partial_sort(pq_ids_distances.begin(), pq_ids_distances.begin() + sort_end, pq_ids_distances.end(),
                 [](Pair &x, Pair &y) -> bool
                 { return x.second < y.second; });

    // output topk labels and distances
    for (int o = 0; o < topk; o++)
    {
      labels[o + i * topk] = o < out_size ? pq_ids_distances[o].first : -1;
      distances[o + i * topk] = o < out_size ? pq_ids_distances[o].second : std::numeric_limits<float>::max();
    }

    pq_ids_distances.clear();
  }
}

int CpuPQ::Remove(int n, int *remove_ids)
{
  for (int i = 0; i < n; i++)
  {
    int id = remove_ids[i];
    auto iter = std::find(std::begin(ids_), std::end(ids_), id);
    if (iter == std::end(ids_))
    {
      std::cout << "CPU PQ remove id " << id << " failed" << std::endl;
      continue;
    }
    else
    {
      int idx = std::distance(std::begin(ids_), iter);
      for (int m_idx = 0; m_idx < M_; m_idx++)
      {
        codes_[(size_t)idx * M_ + m_idx] = codes_[(size_t)(ntotal_ - 1) * M_ + m_idx];
      }
      codes_.erase(std::begin(codes_) + (size_t)(ntotal_ - 1) * M_, std::begin(codes_) + (size_t)ntotal_ * M_);
      ids_[idx] = ids_[ntotal_ - 1];
      ids_.erase(ids_.begin() + ntotal_ - 1);
      ntotal_--;
    }
  }
}

void CpuPQ::SetData(int size, const uint8_t *codes, const int *ids)
{
  codes_.assign(codes, codes + M_ * (size_t)size);
  ids_.assign(ids, ids + size);
  ntotal_ = size;
}

bool test_pq_add(cnindex::PQ * mlu_pq, CpuPQ * cpu_pq, int d, int M, int nbits, int add_num) {
  int ksub = 1 << nbits;

  // prepare random centroids
  std::vector<float> centroids(ksub * d);
  std::default_random_engine e(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> c(-2.0, 2.0);
  for (int i = 0; i < ksub * d; ++i) {
    centroids[i] = c(e);
  }

  // create cnindex pq
  mlu_pq->SetCentroids(centroids.data());

  // create cpu pq
  cpu_pq->SetCentroids(centroids.data());

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
    cpu_pq->Add(add_num, addvecs.data(), ids.data());
    auto ret = mlu_pq->Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "MLU pq.Add() failed" << std::endl;
      return false;
    }

    int m = cpu_pq->M_;
    int ntotal = mlu_pq->GetSize();
    if (ntotal != cpu_pq->ntotal_) {
      std::cout << "ADD failed: mlu ntotal != cpu ntotal" << std::endl;
      return false;
    }

    // compare codes ids
    std::vector<int> mlu_idx(ntotal);
    std::vector<uint8_t> mlu_codes((size_t)ntotal * m);
    std::vector<int> cpu_idx;
    std::vector<uint8_t> cpu_codes;
    mlu_pq->GetData(mlu_codes.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_pq->ids_.data(), cpu_pq->ids_.data() + ntotal);
    cpu_codes.insert(cpu_codes.end(), cpu_pq->codes_.data(), cpu_pq->codes_.data() + (size_t)ntotal * m);

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

    mlu_pq->Reset();
    // prepare add vectors and ids
    std::vector<float> addvecs(add_num * d);
    std::vector<int> ids(add_num);
    for (int i = 0; i < add_num; i++) {
      for (int j = 0; j < d; j++) addvecs[i * d + j] = c(e);
      ids[i] = i;
    }

    int warm_iter_num = 1;
    for (int i = 0; i < warm_iter_num; i++) {
      auto ret = mlu_pq->Add(1, addvecs.data(), ids.data());
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu pq.Add() failed" << std::endl;
        return false;
      }
    }
    mlu_pq->Reset();

    double add_total_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    auto ret = mlu_pq->Add(add_num, addvecs.data(), ids.data());
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

bool test_pq_remove(cnindex::PQ * mlu_pq, CpuPQ * cpu_pq, int nremove, int d, int M, int nbits, int ntotal) {
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
  mlu_pq->SetCentroids(centroids.data());

  // create cpu pq
  cpu_pq->SetCentroids(centroids.data());

  //set data
  cpu_pq->SetData(ntotal, codes.data(), ids.data());
  mlu_pq->SetData(ntotal, codes.data(), ids.data());

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
    cpu_pq->Remove(nremove, remove_ids.data());
    std::cout << "CPU Remove OK" << std::endl;
    auto ret = mlu_pq->Remove(nremove, remove_ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Remove() failed" << std::endl;
      return false;
    }
    std::cout << "MLU Remove OK" << std::endl;
    
    // the list_size after remove
    int size_after_remove = mlu_pq->GetSize();
    if (size_after_remove != cpu_pq->ntotal_) {
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
    mlu_pq->GetData(mlu_codes.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_pq->ids_.data(), cpu_pq->ids_.data() + size_after_remove);
    cpu_codes.insert(cpu_codes.end(), cpu_pq->codes_.data(), cpu_pq->codes_.data() + (size_t)size_after_remove * M);

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
    auto ret = mlu_pq->Reset();
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Reset() failed" << std::endl;
      return false;
    }
    //set data
    mlu_pq->SetData(ntotal, codes.data(), ids.data());
    for (int i = 0; i < warm_iter_num; i++) {
      mlu_pq->Remove(1, remove_ids.data() + i);
    }

    // mlu remove
    double remove_total_time = 0.0;
    ret = mlu_pq->Reset();
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "mlu pq.Reset() failed" << std::endl;
      return false;
    }
    //set data
    mlu_pq->SetData(ntotal, codes.data(), ids.data());

    double test_min_time = std::numeric_limits<double>::max(), test_max_time = 0;
    for (int i = 0; i < nremove; i++) {
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      ret = mlu_pq->Remove(1, remove_ids.data() + i);
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

bool test_flat_add(cnindex::Flat * mlu_flat, CpuFlat * cpu_flat, int d, int add_num) {
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
    cpu_flat->Add(add_num, addvecs.data(), ids.data());
    auto ret = mlu_flat->Add(add_num, addvecs.data(), ids.data());
    if (ret != CNINDEX_RET_SUCCESS) {
      std::cout << "MLU flat.Add() failed" << std::endl;
      return false;
    }

    int ntotal = mlu_flat->GetSize();
    if (ntotal != cpu_flat->ntotal_) {
      std::cout << "ADD failed: mlu ntotal != cpu ntotal" << std::endl;
      return false;
    }

    // compare codes ids
    std::vector<int> mlu_idx(ntotal);
    std::vector<float> mlu_vectors((size_t)ntotal * d);
    std::vector<int> cpu_idx;
    std::vector<float> cpu_vectors;
    mlu_flat->GetData(mlu_vectors.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_flat->ids_.data(), cpu_flat->ids_.data() + ntotal);
    cpu_vectors.insert(cpu_vectors.end(), cpu_flat->xb_.data(), cpu_flat->xb_.data() + (size_t)ntotal * d);

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

    mlu_flat->Reset();
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
      auto ret = mlu_flat->Add(1, addvecs.data(), ids.data());
      if (ret != CNINDEX_RET_SUCCESS) {
        std::cout << "mlu flat.Add() failed" << std::endl;
        return false;
      }
    }
    mlu_flat->Reset();

    double add_total_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    auto ret = mlu_flat->Add(add_num, addvecs.data(), ids.data());
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

bool test_flat_remove(cnindex::Flat * mlu_flat, CpuFlat * cpu_flat, int nremove, int d, int ntotal) {

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
  cpu_flat->Add(ntotal, addvecs.data(), ids.data());
  auto ret = mlu_flat->Add(ntotal, addvecs.data(), ids.data());
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

    int status = mlu_flat->Remove(nremove, remove_vecotr.data());
    int size_after_remove = mlu_flat->GetSize();
    cpu_flat->Remove(nremove, remove_vecotr.data());

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
    mlu_flat->GetData(mlu_vectors.data(), mlu_idx.data());
    cpu_idx.insert(cpu_idx.end(), cpu_flat->ids_.data(), cpu_flat->ids_.data() + num_tmp);
    cpu_vectors.insert(cpu_vectors.end(), cpu_flat->xb_.data(), cpu_flat->xb_.data() + num_tmp * d);

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

    mlu_flat->Reset();
    auto ret = mlu_flat->Add(ntotal, addvecs.data(), ids.data());
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
    ret = mlu_flat->Remove(nremove, remove_vecotr.data());
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
#endif //CPU_PQ_FLAT_CPP