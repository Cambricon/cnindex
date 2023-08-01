#ifndef CN_CPP
#define CN_CPP

#include <cnindex.h>
#include "cpu_pq.hpp"
#include <iostream>
#include <fstream>

using namespace cnindex;
extern "C"
{ // PQ
    PQ *PQ_New(int d, cnindexMetric_t metric, int M, int nbits, int device_id)
    {
        return new PQ(d, metric, M, nbits, device_id);
    }
    void PQ_Del(PQ *pq)
    {
        delete pq;
    }

    void PQ_SetCentroids(PQ *pq, const float *centroids)
    {
        pq->SetCentroids(centroids);
    }

    void PQ_Add(PQ *pq, int n_add, const float *x, const int *idx)
    {
        auto status = pq->Add(n_add, x, idx);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "pq.Add() failed" << std::endl;
        }
        else
        {
            std::cout << "pq.Add() sucess" << std::endl;
        }
    }

    void PQ_Search(PQ *pq, int n_search, const float *x, int topk, int *labels, float *distances)
    {
        auto status = pq->Search(n_search, x, topk, labels, distances);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "pq.Search() failed" << std::endl;
        }
        else
        {
            std::cout << "pq.Search() sucess" << std::endl;
        }
    }
    void PQ_Remove(PQ *pq, int n_remove, int *remove_ids)
    {
        auto status = pq->Remove(n_remove, remove_ids);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "pq.Remove() failed" << std::endl;
        }
        else
        {
            std::cout << "pq.Remove() sucess" << std::endl;
        }
    }

    int PQ_GetSize(PQ *pq)
    {
        return pq->GetSize();
    }

    void PQ_GetData(PQ *pq, uint8_t* codes, int *ids)
    {
        pq->GetData(codes, ids);
    }

    void PQ_Reset(PQ *pq)
    {
        auto status = pq->Reset();
        int size_after_reset = pq->GetSize();
        if (status != CNINDEX_RET_SUCCESS && size_after_reset != 0)
        {
            std::cout << "pq.Reset failed" << std::endl;
        }
        else
        {
            std::cout << "pq.Reset sucess! after reset size = " << size_after_reset << std::endl;
        }
    }

    void PQ_SetData(PQ *pq, int ntotal, const uint8_t *codes, const int *ids)
    {
        pq->SetData(ntotal, codes, ids);
    }
}

extern "C"
{
    CpuPQ *CPU_PQ_New(int d, int M, int nbits)
    {
        return new CpuPQ(d, M, nbits);
    }

    void CPU_PQ_Del(CpuPQ *cpu_pq)
    {
        delete cpu_pq;
    }

    void CPU_PQ_SetCentroids(CpuPQ *cpu_pq, const float *centroids)
    {
        cpu_pq->SetCentroids(centroids);
    }

    void CPU_PQ_Add(CpuPQ *cpu_pq, int n_add, const float *x, const int *idx)
    {
        cpu_pq->Add(n_add, x, idx);
    }

    void CPU_PQ_Search(CpuPQ *cpu_pq, int n_search, float *x, int topk, int *labels, float *distances)
    {
        cpu_pq->Search(n_search, x, topk, labels, distances);
        std::cout << "CPU Search OK" << std::endl;
    }

    void CPU_PQ_Remove(CpuPQ *cpu_pq, int n_remove, int *remove_ids)
    {
        cpu_pq->Remove(n_remove, remove_ids);
    }

    void CPU_PQ_SetData(CpuPQ *cpu_pq, int ntotal, const uint8_t *codes, const int *ids)
    {
        cpu_pq->SetData(ntotal, codes, ids);
    }

    void CPU_PQ_Reset(CpuPQ *cpu_pq)
    {
        cpu_pq->Reset();
    }

    int CPU_PQ_GetnTotal(CpuPQ *cpu_pq)
    {
        return cpu_pq->ntotal_;
    }
    
    int CPU_PQ_GetM_(CpuPQ *cpu_pq)
    {
        return cpu_pq->M_;
    }
}

extern "C"
{
    void Test_Add(PQ* mlu_pq, CpuPQ* cpu_pq, int d, int M, int nbits, int add_num)
    {
        test_add(mlu_pq, cpu_pq, d, M, nbits, add_num);
    }
    // void test_Search(PQ * mlu_pq, CpuPQ* cpu_pq, int d, int M, int nbits, int add_num )
    // {
    //     test_search(mlu_pq, cpu_pq, d, M, nbits, add_num);
    // }
    void Test_Remove(PQ* mlu_pq, CpuPQ* cpu_pq, int nremove, int d, int M, int nbits, int ntotal)
    {
        test_remove(mlu_pq, cpu_pq, nremove, d, M, nbits, ntotal);
    }
}
#endif // CN_CPP