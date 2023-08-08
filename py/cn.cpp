#ifndef CN_CPP
#define CN_CPP

#include <cnindex.h>
#include "cpu_pq_flat.hpp"
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
    CpuPQ *Cpu_PQ_New(int d, int M, int nbits)
    {
        return new CpuPQ(d, M, nbits);
    }

    void Cpu_PQ_Del(CpuPQ *cpu_pq)
    {
        delete cpu_pq;
    }

    void Cpu_PQ_SetCentroids(CpuPQ *cpu_pq, const float *centroids)
    {
        cpu_pq->SetCentroids(centroids);
    }

    void Cpu_PQ_Add(CpuPQ *cpu_pq, int n_add, const float *x, const int *idx)
    {
        cpu_pq->Add(n_add, x, idx);
    }

    void Cpu_PQ_Search(CpuPQ *cpu_pq, int n_search, float *x, int topk, int *labels, float *distances)
    {
        cpu_pq->Search(n_search, x, topk, labels, distances);
        std::cout << "Cpu Search OK" << std::endl;
    }

    void Cpu_PQ_Remove(CpuPQ *cpu_pq, int n_remove, int *remove_ids)
    {
        cpu_pq->Remove(n_remove, remove_ids);
    }

    void Cpu_PQ_SetData(CpuPQ *cpu_pq, int ntotal, const uint8_t *codes, const int *ids)
    {
        cpu_pq->SetData(ntotal, codes, ids);
    }

    void Cpu_PQ_Reset(CpuPQ *cpu_pq)
    {
        cpu_pq->Reset();
    }

    int Cpu_PQ_GetnTotal(CpuPQ *cpu_pq)
    {
        return cpu_pq->ntotal_;
    }
    
    int Cpu_PQ_GetM_(CpuPQ *cpu_pq)
    {
        return cpu_pq->M_;
    }
}
extern "C"
{
    //PQ test add and remove
    void Test_PQ_Add(PQ* mlu_pq, CpuPQ* cpu_pq, int d, int M, int nbits, int add_num)
    {
        test_pq_add(mlu_pq, cpu_pq, d, M, nbits, add_num);
    }

    void Test_PQ_Remove(PQ* mlu_pq, CpuPQ* cpu_pq, int nremove, int d, int M, int nbits, int ntotal)
    {
        test_pq_remove(mlu_pq, cpu_pq, nremove, d, M, nbits, ntotal);
    }
 
    // Flat
    Flat *Flat_New(int d, cnindexMetric_t metric, int device_id)
    {
        return new Flat(d, metric, device_id);
    }

    void Flat_Del(Flat *flat)
    {
        delete flat;
    }

    void Flat_Add(Flat *flat, int n_add, float *x, int *idx)
    {
        auto status = flat->Add(n_add, x, idx);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "flat.Add() failed" << std::endl;
        }
        else
        {
            std::cout << "flat.Add() sucess" << std::endl;
        }
    }

    void Flat_Search(Flat *flat, int n_search, const float *x, int topk, int *labels, float *distances)
    {
        auto status = flat->Search(n_search, x, topk, labels, distances);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "flat.Search() failed" << std::endl;
        }
        else
        {
            std::cout << "pq.Search() sucess" << std::endl;
        }
    }

    void Flat_Remove(Flat *flat, int n_remove, int *remove_ids)
    {
        auto status = flat->Remove(n_remove, remove_ids);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "flat.Remove() failed" << std::endl;
        }
        else
        {
            std::cout << "flat.Remove() sucess" << std::endl;
        }
    }

    int Flat_GetSize(Flat *flat)
    {
        return flat->GetSize();
    }

    void Flat_GetData(Flat *flat, float* codes, int *ids)
    {
        flat->GetData(codes, ids);
    }

    void Flat_Reset(Flat *flat)
    {
        auto status = flat->Reset();
        int size_after_reset = flat->GetSize();
        if (status != CNINDEX_RET_SUCCESS && size_after_reset != 0)
        {
            std::cout << "flat.Reset failed" << std::endl;
        }
        else
        {
            std::cout << "flat.Reset sucess! after reset size = " << size_after_reset << std::endl;
        }
    }
}
extern "C"
{
    // Cpu_Flat
    CpuFlat *Cpu_Flat_New(int d, cnindexMetric_t metric)
    {
        return new CpuFlat(d, metric);
    }

    void Cpu_Flat_Del(CpuFlat *cpu_flat)
    {
        delete cpu_flat;
    }

    void Cpu_Flat_Add(CpuFlat *cpu_flat, int n_add, float *x, int *idx)
    {
        auto status = cpu_flat->Add(n_add, x, idx);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "cpu_flat.Add() failed" << std::endl;
        }
        else
        {
            std::cout << "cpu_flat.Add() sucess" << std::endl;
        }
    }

    void Cpu_Flat_Search(CpuFlat *cpu_flat, int n_search, const float *x, int topk, int *labels, float *distances)
    {
        auto status = cpu_flat->Search(n_search, x, topk, labels, distances);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "cpu_flat.Search() failed" << std::endl;
        }
        else
        {
            std::cout << "cpu_flat.Search() sucess" << std::endl;
        }
    }
    
    void Cpu_Flat_Remove(CpuFlat *cpu_flat, int n_remove, int *remove_ids)
    {
        auto status = cpu_flat->Remove(n_remove, remove_ids);
        if (status != CNINDEX_RET_SUCCESS)
        {
            std::cout << "flat.Remove() failed" << std::endl;
        }
        else
        {
            std::cout << "flat.Remove() sucess" << std::endl;
        }
    }

    int Cpu_Flat_GetnTotal(CpuFlat *cpu_flat)
    {
        return cpu_flat->ntotal_;
    }
    
    // Flat test add and remove
    void Test_Flat_Add(Flat* mlu_flat, CpuFlat* cpu_flat, int d, int add_num)
    {
        test_flat_add(mlu_flat, cpu_flat, d, add_num);
    }

    void Test_Flat_Remove(Flat* mlu_flat, CpuFlat* cpu_flat, int nremove, int d, int ntotal)
    {
        test_flat_remove(mlu_flat, cpu_flat, nremove, d, ntotal);
    }
}
#endif // CN_CPP