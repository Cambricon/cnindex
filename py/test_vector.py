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

def test_search(device_id, mode_set, nq, d, M, nbits, ntotal, topk, vector):
    ksub = 1 << nbits
    code_size = (nbits * M + 7) / 8

    # prepare random centroids codes ids
    centroids = np.random.uniform(0, 0, ksub * d)
    # codes = np.random.randint(0, 256, int(ntotal * code_size), dtype=np.uint8)
    # ids_set = np.arange(0, ntotal) 

    # create cnindex pq
    # pq = PQ(d, cnindexMetric_t.CNINDEX_METRIC_L2, M, nbits, device_id)
    # pq.SetCentorids(centroids)     

    # create cpu pq
    cpu_pq = CPU_PQ(d, M, nbits) 
    cpu_pq.SetCentorids(centroids)

    # add vector
    ids = np.arange(0, 1)
    # print(f"nadd: {n_add}, addvecs: {addvecs.shape}, ids: {ids.shape}")
    # print("vector:",vector,len(vector))
    # print("ids:",ids)

    # set data
    # pq.SetData(ntotal, vector, ids)
    # cpu_pq.SetData(ntotal, vector, ids) 
    # pq.Add(1, vector, ids)
    cpu_pq.Add(1, vector, ids)

    # query vector
    search_data = vector

    print("-------------- PQ SEARCH ACCURACY TEST --------------\n")
    # print("Search dataset:  " + "random" if mode_set == "s" else mode_set)
    print("       nquery :  ", nq)
    print("       d      :  ", d)
    print("       M      :  ", M)
    print("       nbits  :  ", nbits)
    print("       ntotal :  ", ntotal)
    print("       topk   :  ", topk)
    
    # search accuracy
    # mlu_labels, mlu_distances = pq.Search(nq, search_data, topk)
    # cpu_labels, cpu_distances = cpu_pq.Search(nq, search_data, topk) 
    # # print(search_data[:100])
    # print(cpu_distances)
    # print(cpu_labels)
    # distances_mae, distances_mse = compare_mae_mse(cpu_distances, mlu_distances, nq ,topk)
    # labels_diff = compare(cpu_labels, mlu_labels, nq, topk, True, True)
    
    # print("Diff labels: {}%".format(labels_diff * 100))
    # print("distances_mae:{}%".format(distances_mae * 100))
    # print("distances_mse:{}%".format(distances_mse * 100))

if __name__ == '__main__':
  
    device_id = 0
    nq = 1
    d = 256
    M = 32
    nbits = 8
    ntotal = 1
    topk = 1

    xb = np.random.random((1, d)).astype('float32')
    # print(xb)
    norm = np.linalg.norm(xb, axis=1, keepdims=True)
    vector = xb / norm
    vector = vector.reshape(-1, )
    # vector[:32]=0
    # vector[32:64]=1
    # vector[64:128]=2
    # vector[128: 160]=3
    # vector[160: 192]=4
    # vector[192: 224]=5
    # vector[224: 256]=6

    # print(vector,len(vector))
    # vector = np.zeros_like(vector)

    test_search(device_id, 's', nq, d, M, nbits, ntotal, topk, vector)

