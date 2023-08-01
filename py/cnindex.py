import ctypes
from ctypes import *
import numpy as np
import sys

class cnindexMetric_t:
    CNINDEX_METRIC_L1 = 0
    CNINDEX_METRIC_L2 = 0
    CNINDEX_METRIC_IP = 1
    
class cnindexReturn_t:
    CNINDEX_RET_SUCCESS         = 0
    CNINDEX_RET_NOT_IMPL        = -1
    CNINDEX_RET_NOT_VALID       = -2 
    CNINDEX_RET_BAD_PARAMS      = -3
    CNINDEX_RET_ALLOC_FAILED    = -4 
    CNINDEX_RET_OP_FAILED       = -5

ll = ctypes.cdll.LoadLibrary
ctypes.CDLL("/usr/local/neuware/lib64/libcnrt.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("/usr/local/neuware/lib64/libcnnl.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("/usr/local/neuware/lib64/libcnnl_extra.so", mode=ctypes.RTLD_GLOBAL)
api = ll("../lib/libcnindex.so")

class PQ:
    def __init__(self, d_ , Metric_t , M_ , nbits_, device_id_):
        api.PQ_Add.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_float), POINTER(c_int))  
        api.PQ_Del.argtypes = (ctypes.c_void_p,) 
        api.PQ_GetData.argtypes = (ctypes.c_void_p, POINTER(c_uint8), POINTER(c_int)) 
        api.PQ_GetSize.argtypes = (ctypes.c_void_p,) 
        api.PQ_GetSize.restype = ctypes.c_int
        api.PQ_New.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,ctypes.c_int,ctypes.c_int)
        api.PQ_New.restype = ctypes.c_void_p
        api.PQ_Remove.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_int)) 
        api.PQ_Reset.argtypes = (ctypes.c_void_p,)
        api.PQ_Search.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_float), ctypes.c_int, POINTER(c_int), POINTER(c_float))  
        api.PQ_SetCentroids.argtypes = (ctypes.c_void_p, POINTER(c_float)) 
        api.PQ_SetData.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_uint8), POINTER(c_int))
        self._ptr = api.PQ_New(d_, Metric_t, M_, nbits_, device_id_)

    def SetCentorids(self, centroids_data):
        centroids_array = (ctypes.c_float * len(centroids_data))(*centroids_data) 
        api.PQ_SetCentroids(self._ptr, centroids_array)

    def Add(self, n_add, addvecs, ids):
        addvecs_array = (ctypes.c_float * len(addvecs))(*addvecs) 
        ids_array = (ctypes.c_int * len(ids))(*ids) 
        api.PQ_Add(self._ptr, n_add, addvecs_array, ids_array)

    def Search(self, n_search, data, topk):
        labels = np.zeros(n_search * topk, dtype = int)
        distances = np.zeros(n_search * topk, dtype = float) 
        data_array = (ctypes.c_float * len(data))(*data) 
        labels_array = (ctypes.c_int * len(labels))(*labels) 
        distances_array = (ctypes.c_float * len(distances))(*distances) 
        api.PQ_Search(self._ptr, n_search, data_array, topk, labels_array, distances_array)
        return np.array(labels_array),np.array(distances_array) 

    def GetSize(self):
        return api.PQ_GetSize(self._ptr) 
    
    def GetData(self, ntotal, m):
        mlu_idx = np.zeros(ntotal, dtype = int)
        mlu_codes = np.zeros(ntotal * m, dtype= c_uint8)
        mlu_idx_array = (ctypes.c_uint8 * len(mlu_idx))(*mlu_idx) 
        mlu_codes_array = (ctypes.c_int * len(mlu_codes))(*mlu_codes) 
        api.PQ_GetData(self._ptr, mlu_codes_array, mlu_idx_array)
        return np.array(mlu_codes_array), np.array(mlu_idx_array) 
    
    def Remove(self, n_remove, remove_ids):
        remove_ids_array = (ctypes.c_int * len(remove_ids))(*remove_ids) 
        api.PQ_Remove(self._ptr, n_remove, remove_ids_array)
        size_after_remove = self.GetSize() 
        print("size_after_remove:",size_after_remove)
  
    def SetData(self, ntotal, codes, ids):
        codes_array = (ctypes.c_uint8 * len(codes))(*codes) 
        ids_array = (ctypes.c_int * len(ids))(*ids) 
        api.PQ_SetData(self._ptr, ntotal, codes_array, ids_array)

    def Reset(self):
        api.PQ_Reset(self._ptr)
        
    def __del__(self):
        api.PQ_Del(self._ptr)

class CPU_PQ:
    def __init__(self, d_, M_, nbits_):
        api.CPU_PQ_Add.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_float), POINTER(c_int))  
        api.CPU_PQ_Del.argtypes = (ctypes.c_void_p,) 
        api.CPU_PQ_New.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
        api.CPU_PQ_New.restype = ctypes.c_void_p
        api.CPU_PQ_Remove.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_int)) 
        api.CPU_PQ_Reset.argtypes = (ctypes.c_void_p,)
        api.CPU_PQ_Search.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_float), ctypes.c_int, POINTER(c_int), POINTER(c_float))  
        api.CPU_PQ_SetCentroids.argtypes = (ctypes.c_void_p, POINTER(c_float)) 
        api.CPU_PQ_SetData.argtypes = (ctypes.c_void_p, ctypes.c_int, POINTER(c_uint8), POINTER(c_int))

        self._ptr = api.CPU_PQ_New(d_, M_, nbits_)
        self._M = M_

    def SetCentorids(self, centroids_data):
        centroids_array = (ctypes.c_float * len(centroids_data))(*centroids_data) 
        api.CPU_PQ_SetCentroids(self._ptr, centroids_array)

    def Add(self, n_add, addvecs, ids):
        addvecs_array = (ctypes.c_float * len(addvecs))(*addvecs) 
        ids_array = (ctypes.c_int * len(ids))(*ids) 
        api.CPU_PQ_Add(self._ptr, n_add, addvecs_array, ids_array)

    def Search(self, n_search, data, topk):
        labels = np.zeros(n_search * topk, dtype = int)
        distances = np.zeros(n_search * topk, dtype = float) 
        data_array = (ctypes.c_float * len(data))(*data) 
        labels_array = (ctypes.c_int * len(labels))(*labels) 
        distances_array = (ctypes.c_float * len(distances))(*distances) 
        api.CPU_PQ_Search(self._ptr, n_search, data_array, topk, labels_array, distances_array)
        return np.array(labels_array),np.array(distances_array), 
    
    def Remove(self, n_remove, remove_ids):
        remove_ids_array = (ctypes.c_int * len(remove_ids))(*remove_ids) 
        api.CPU_PQ_Remove(self._ptr, n_remove, remove_ids_array)

    def SetData(self, ntotal, codes, ids):
        codes_array = (ctypes.c_uint8 * len(codes))(*codes) 
        ids_array = (ctypes.c_int * len(ids))(*ids) 
        api.CPU_PQ_SetData(self._ptr, ntotal, codes_array, ids_array)

    def Reset(self):
        api.CPU_PQ_Reset(self._ptr)
        
    def GetnTotal(self):
        api.CPU_PQ_GetnTotal.argtypes = (ctypes.c_void_p)
        api.CPU_PQ_GetnTotal.restype = ctypes.c_int
        return api.CPU_PQ_GetnTotal(self._ptr)
    
    def GetM_(self):
        api.CPU_PQ_GetM_.argtypes = (ctypes.c_void_p)
        api.CPU_PQ_GetM_.restype = ctypes.c_int
        return api.CPU_PQ_GetM_(self._ptr) 
    
    def __del__(self):
        api.CPU_PQ_Del(self._ptr)