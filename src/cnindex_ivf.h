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

#ifndef __CNINDEX_IVF_H__
#define __CNINDEX_IVF_H__

#include <stdint.h>
#include <vector>

#include "cnindex.h"

namespace cnindex {

class Flat;

namespace impl {

/*!
 * @class IVF
 *
 * @brief IVF is the base class of indices using Invert File System to store vectors.
 * It defines APIs and members for derived class of indices.
 *
 * @note - None.
 *
 * @par Requirements
 * - cnindex.h
 *
 */
class IVF {
 public:
  /**
   * @brief Constructor.
   *
   * @param[in] flat The memory pointer of flat index for coarse search.
   * @param[in] metric The metric type of distance computation.
   * @param[in] vector_size The vector size in bytes.
   * @param[in] device_id The device to be operated on.
   *
   */
  IVF(const cnindex::Flat *flat, cnindexMetric_t metric, int vector_size, int device_id);

  /**
   * @brief Destructor.
   */
  virtual ~IVF();

  /**
   * @brief Resets index.
   *
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t Reset();

  /**
   * @brief Sets vectors to an IVF list.
   *
   * @param[in] index The index of IVF list.
   * @param[in] size The number of vectors to set.
   * @param[in] vectors The memory pointer stored code of vectors to set, size: list_size * vector_size.
   * @param[in] ids The memory pointer stored id of vectors to set, size: list_size.
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t SetListData(int index, int size, const void *vectors, const int *ids);

  /**
   * @brief Gets the number of vectors in an IVF list.
   *
   * @param[in] index The index of IVF list.
   * @retval >=0 The number of vectors.
   *         Otherwise, the error code is returned.
   */
  virtual int GetListSize(int index) const;

  /**
   * @brief Gets vectors from an IVF list.
   *
   * @param[in] index The index of IVF list.
   * @param[out] vectors The memory pointer to store code of vectors, size: list_size * vector_size.
   * @param[out] ids The memory pointer to store id of vectors, size: list_size.
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t GetListData(int index, void *vectors, int *ids) const;

  /**
   * @brief Searchs vectors in index.
   *
   * @param[in] n The number of input vectors to search.
   * @param[in] x The memory pointer of input vectors to search, size: n * d.
   * @param[in] nprobe The probe number of IVF list.
   * @param[in] k The TopK of searching result.
   * @param[out] ids The memory pointer to save result of ids, size: n * k.
   * @param[out] distances The memory pointer to save result of distances, size: n * k.
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t Search(int n, const float *x, int nprobe, int k, int *ids, float *distances) const;

  /**
   * @brief Adds vectors index.
   *
   * @param[in] n The number of vectors to add.
   * @param[in] x The memory pointer of vectors to add, size: n * d.
   * @param[in] ids The memory pointer of id of adding vectors, size: n.
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t Add(int n, const float *x, const int *ids);

  /**
   * @brief Removes vectors from index.
   *
   * @param[in] n The number of vectors to remove.
   * @param[in] ids The memory pointer of id of removing vectors, size: n.
   * @retval CNINDEX_RET_SUCCESS The function ends normally.
   *         Otherwise, the error code is returned.
   */
  virtual cnindexReturn_t Remove(int n, const int *ids);

 protected:
  const cnindex::Flat *flat_;

  const int d_;
  const cnindexMetric_t metric_;
  const int nlist_;
  const int vector_size_;
  int ntotal_;

  const int device_id_;

  std::vector<int> nlist_size_;
  std::vector<void *> vectors_ptr_;
  std::vector<void *> ids_ptr_;
};  // IVF

}  // impl

}  // cnindex

#endif  // __CNINDEX_IVF_H__
