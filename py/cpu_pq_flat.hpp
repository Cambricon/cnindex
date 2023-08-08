#ifndef CPU_PQ_FLAT_HPP
#define CPU_PQ_FLAT_HPP
#include "cnindex.h"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include <cstring>
#include <fstream>

class CpuPQ
{
public:
  CpuPQ(int d, int M, int nbits) : d_(d), M_(M), nbits_(nbits)
  {
    dsub_ = d_ / M_;
    ksub_ = 1 << nbits_;
    ntotal_ = 0;
    code_size_ = (nbits * M + 7) / 8;
    codes_.clear();
    ids_.clear();
  }

  ~CpuPQ() {}

  int Add(int n, const float *x, const int *ids);
  int Search(int n, float *x, int topk, int *labels, float *distances);
  int Remove(int n, int *remove_ids);
  void SetData(int size, const uint8_t *codes, const int *ids);
  int Reset()
  {
    codes_.clear();
    ids_.clear();
    ntotal_ = 0;
  }
  int SetCentroids(const float *centroids)
  {
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

bool test_flat_add(cnindex::Flat * mlu_flat, CpuFlat * cpu_flat, int d, int add_num);
bool test_flat_remove(cnindex::Flat * mlu_flat, CpuFlat * cpu_falt, int nremove, int d, int ntotal);
bool test_pq_add(cnindex::PQ * mlu_pq, CpuPQ * cpu_pq, int d, int M, int nbits, int add_num);
bool test_pq_remove(cnindex::PQ * mlu_pq, CpuPQ * cpu_pq, int nremove, int d, int M, int nbits, int ntotal);

#endif //CPU_PQ_FLAT_HPP