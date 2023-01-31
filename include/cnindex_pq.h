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

#ifndef __CNINDEX_PQ_H__
#define __CNINDEX_PQ_H__

#include <stdint.h>

#include "cnindex.h"

namespace cnindex {

namespace impl { class PQ; }

/*!
 * @class PQ
 *
 * @brief PQ is a class of Product Quantization index.
 *        It offers the ability to manage a pure (without IVF) product quantization based on vector database.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - cnindex.h
 *
 * @par Example
 * - See sample codes in samples/sample_cnindex_pq.cpp.
 *
 */
class PQ {
 public:
  /**
   * @brief Constructor.
   *
   * @param[in] d The dimensionality of vectors.
   * @param[in] metric The metric type of distance computation.
   * @param[in] M The subquantizers number of PQ encoding.
   * @param[in] nbits The bit number of PQ encoding.
   * @param[in] device_id The ID of the device to perform operations.
   *
   */
  PQ(int d, cnindexMetric_t metric, int M, int nbits, int device_id);

  /**
   * @brief Destructor.
   *
   */
  ~PQ();

  /**
   * @brief Deleted copy constructor.
   */
  PQ(const PQ&) = delete;

  /**
   * @brief Deleted copy overloading assign operator.
   */
  PQ &operator=(const PQ&) = delete;

  /**
   * @brief Move constructor.
   */
  PQ(PQ&&);

  /**
   * @brief Move overloading assign operator.
   */
  PQ &operator=(PQ&&);

  /**
   * @brief Resets a PQ index. This function clears all vectors in the PQ index except centroids.
   *
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t Reset();

  /**
   * @brief Sets centroids to a PQ index.
   *
   * @param[in] centroids The centroids of product quantization, the size of centroids equals ( \b 1 << \b nbits) * \b d ,
   *                      where \b nbits is the bits number of PQ encoding and \b d is the dimensionality of vector.
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t SetCentroids(const float *centroids);

  /**
   * @brief Sets encoded vectors to a PQ index.
   *
   * @param[in] size The number of vectors to set.
   * @param[in] codes Pointer to the host memory that stores the code of vectors.
   *                  The size of codes equals \b size * \b d ,
   *                  where \b d is the dimensionality of vector.
   * @param[in] ids Pointer to the host memory that stores the IDs of vectors to set.
   *                The size of IDs equals parameter \b size .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t SetData(int size, const uint8_t *codes, const int *ids);

  /**
   * @brief Gets the number of vectors in a PQ index.
   *
   * @retval
   * - >=0 The number of vectors.
   * - <0 The error code is returned.
   */
  int GetSize() const;

  /**
   * @brief Gets encoded vectors from a PQ index.
   *
   * @param[out] codes Pointer to the host memory that stores the code of vectors.
   *                   The size of codes equals \b ntotal * \b code_size ,
   *                   where \b ntotal is the total number of vectors in a PQ index
   *                   and \b code_size is the bytes number of PQ encoded vector.
   * @param[out] ids Pointer to the host memory that stores the IDs of vectors.
   *                 The size of IDs equals \b ntotal .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t GetData(uint8_t *codes, int *ids) const;

  /**
   * @brief Searchs vectors in a PQ index.
   *
   * @param[in] n The number of input vectors to search.
   * @param[in] x Pointer to the host memory that stores the input vectors to search.
   *              The size of vectors equals \b n * \b d, where \b d is the dimensionality of vector.
   * @param[in] k The TopK for searching.
   * @param[out] ids Pointer to the host memory that stores the IDs of the searching results.
   *                 The size of IDs equals \b n * \b k .
   * @param[out] distances Pointer to the host memory that stores the distances of the searching results.
   *                       The size of distances equals \b n * \b k .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances = nullptr) const;

  /**
   * @brief Adds vectors to a PQ index.
   *
   * @param[in] n The number of vectors to add.
   * @param[in] x Pointer to the host memory that stores the input vectors to add.
   *              The size of vectors equals \b n * \b d, where \b d is the dimensionality of vector.
   * @param[in] ids Pointer to the host memory that stores the IDs of the adding vectors.
   *                The size of IDs equals \b n .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t Add(int n, const float *x, const int *ids = nullptr);

  /**
   * @brief Removes vectors from a PQ index.
   *
   * @param[in] n The number of vectors to remove.
   * @param[in] ids  Pointer to the host memory that stores the IDs of the removing vector.
   *                 The size of IDs equals \b n .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   */
  cnindexReturn_t Remove(int n, const int *ids);

 private:
  impl::PQ *pq_ = nullptr;
};  // PQ

}  // cnindex

#endif  // __CNINDEX_PQ_H__
