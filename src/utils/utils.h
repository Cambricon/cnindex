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

#ifndef __CNINDEX_UTILS_H__
#define __CNINDEX_UTILS_H__

#include <stdint.h>

#include <cnrt.h>
#include <cnnl_extra.h>

#include "thread_pool.h"

#if (CNRT_MAJOR_VERSION < 5 && CNNL_EXTRA_MAJOR == 0 && CNNL_EXTRA_MINOR <= 2)
  #define ENABLE_MLU200 1
#elif (CNRT_MAJOR_VERSION >= 5)
#if (CNNL_EXTRA_MAJOR==0 && CNNL_EXTRA_MINOR < 18)
  #error "CNNL Extra version must >= 0.18"
#else
  #define ENABLE_MLU300 1
#if CNRT_MAJOR_VERSION == 5
  #define cnrtQueueCreate cnrtCreateQueue
  #define cnrtQueueSync cnrtSyncQueue
  #define cnrtQueueDestroy cnrtDestroyQueue
#endif
#endif
#endif

#define USE_BFC_ALLOCATOR
#define USE_THREAD_POOL

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define ALIGN_128(x) ALIGN(x, 128)

namespace cnindex {

void CNRTInit();
void DeviceGuard(int device_id);
void * AllocMLUMemory(size_t bytes, int device_id = -1);
void FreeMLUMemory(void *ptr, int device_id = -1);
uint32_t CeilPower2(uint32_t n);
uint32_t FloorPower2(uint32_t n);
size_t GetCPUCoreNumber();
#ifdef USE_THREAD_POOL
EqualityThreadPool * GetThreadPool();
#endif

}  // namespace cnindex

#endif  // __CNINDEX_UTILS_H__
