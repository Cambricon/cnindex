import ctypes
from ctypes import *
import numpy as np
import sys
import argparse
from cnindex import PQ, CPU_PQ, api
from cnindex import cnindexMetric_t, cnindexReturn_t

def compare(ref, test, batch, size, s, p):
    diff = 0
    total = batch * size
    if s == True:
        for i in range(batch):
            if ref[(i + 1) * size - 1] != test[(i + 1) * size - 1] and \
               (not(ref[(i + 1) * size - 2] == test[(i + 1) * size -1] \
                and ref[(i + 1) * size -1] == test[(i + 1) * size - 2])):
                ref[i * size : (i + 1) * size - 1].sort()
                test[i * size : (i + 1) * size - 1].sort()
            else:
                ref[i * size : (i + 1) * size].sort()
                test[i * size : (i + 1) * size].sort() 

    for i in range(batch):
        error = False
        for j in range(size):
            if ref[i * size + j] != test[i * size + j]:
                diff = diff + 1
                error = True
        if p and error:
            print("ref[{}]".format(i))
            print(ref[i * size + j] for j in range(size))
            print("test[{}]".format(i))
            print(test[i * size + j] for j in range(size))
    
    return float(diff/total)

def compare_mae_mse(ref, test, batch, size):
    if len(ref) != len(test):
        print("compair feature not match")
        return -1
    total = batch * size
    mae = sum(abs(a-b) for a, b in zip(ref,test))
    mse = sum(pow(a-b,2) for a,b in zip(ref,test))
    return mae / total, mse / total

def get_args():
    # parser argument
    parser = argparse.ArgumentParser(description="CnIndex test work")
    parser.add_argument("--device_id", default = 0, type=int,
                        help="device number run model")
    parser.add_argument("--mode_set", default = "s", type=str,
                        help="search(s), add(a), remove(r) mode.")
    parser.add_argument("--nq", default = 1, type=int,
                        help="number of query")
    parser.add_argument("--d", default= 256, type=int,
                        help="输入向量维度dim限制为256、512、1024")
    parser.add_argument("--M", default= 32, type=int,
                        help="PQ量化的向量子空间个数 M 限制为32、64")
    parser.add_argument("--nbits", default = 8, type=int,
                        help="输入向量分解成每个低维向量的存储位数 nbits 限制为 8")
    parser.add_argument("--ntotal", default = 400, type=int,
                        help="dataset's size")
    parser.add_argument("--topk", default = 1, type=int,
                        help="topk num")
    return parser.parse_args()

def test_search(device_id, mode_set, nq, d, M, nbits, ntotal, topk):
    ksub = 1 << nbits
    code_size = (nbits * M + 7) / 8

    # prepare random centroids codes ids
    centroids = np.random.uniform(-2.0, 2.0, ksub * d)

    codes = np.random.randint(0, 256, int(ntotal * code_size), dtype=np.uint8)
    ids_set = np.arange(0, ntotal) 

    # create cnindex pq
    pq = PQ(d, cnindexMetric_t.CNINDEX_METRIC_L2, M, nbits, device_id)
    pq.SetCentorids(centroids)     

    # create cpu pq
    cpu_pq = CPU_PQ(d, M, nbits) 
    cpu_pq.SetCentorids(centroids)

    # set data
    pq.SetData(ntotal, codes, ids_set)
    cpu_pq.SetData(ntotal, codes, ids_set) 

    # query vector
    search_data = np.random.uniform(-3.0, 3.0, nq * d)

    print("-------------- PQ SEARCH ACCURACY TEST --------------\n")
    print("Search dataset:  " + "random" if mode_set == "s" else mode_set)
    print("       nquery :  ", nq)
    print("       d      :  ", d)
    print("       M      :  ", M)
    print("       nbits  :  ", nbits)
    print("       ntotal :  ", ntotal)
    print("       topk   :  ", topk)

    # search accuracy
    mlu_labels, mlu_distances = pq.Search(nq, search_data, topk)
    cpu_labels, cpu_distances = cpu_pq.Search(nq, search_data, topk) 
    distances_mae, distances_mse = compare_mae_mse(cpu_distances, mlu_distances, nq ,topk)
    labels_diff = compare(cpu_labels, mlu_labels, nq, topk, True, True)
    
    print("Diff labels: {}%".format(labels_diff * 100))
    print("distances_mae:{}%".format(distances_mae * 100))
    print("distances_mse:{}%".format(distances_mse * 100))
    
def test_add(device_id, d, M, nbits, add_num):
    pq = PQ(d, cnindexMetric_t.CNINDEX_METRIC_L2, M, nbits, device_id)
    cpu_pq = CPU_PQ(d, M, nbits) 
    api.Test_Add.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)  
    api.Test_Add(pq._ptr, cpu_pq._ptr, d, M, nbits, add_num)

def test_remove(device_id, nremove, d, M, nbits, ntotal):
    pq = PQ(d, cnindexMetric_t.CNINDEX_METRIC_L2, M, nbits, device_id)
    cpu_pq = CPU_PQ(d, M, nbits) 
    api.Test_Remove.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)  
    api.Test_Remove(pq._ptr, cpu_pq._ptr, nremove, d, M, nbits, ntotal)

if __name__ == '__main__':
    args = get_args()

    device_id = args.device_id
    mode_set = args.mode_set
    nq = args.nq 
    d = args.d
    M = args.M
    nbits = args.nbits
    ntotal = args.ntotal
    topk = args.topk

    if (mode_set == "a"):
        test_add(device_id, d, M, nbits, ntotal)
    elif (mode_set == "r"):
        test_remove(device_id, nq, d, M, nbits, ntotal)
    else:
        test_search(device_id, mode_set, nq, d, M, nbits, ntotal, topk)
