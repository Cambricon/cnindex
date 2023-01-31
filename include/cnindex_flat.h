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

#ifndef __CNINDEX_FLAT_H__
#define __CNINDEX_FLAT_H__

#include "cnindex.h"

namespace cnindex {

namespace impl {
   class Flat;
   class IVFPQ2;
   class IVFPQ3;
}

/*!
 * @class Flat
 *
 * @brief Flat is a class of flat type index which is used to store the full vectors and perform exhaustive search.
 *        It offers the ability to manage index based on vector database.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - cnindex.h
 *
 * @par Example
 * - See sample codes in samples/sample_cnindex_flat.cpp.
 *
 */
class Flat {
 public:
  /**
   * @brief Constructor.
   *
   * @param[in] d The dimensionality of vectors.
   * @param[in] metric The metric type of distance computation.
   * @param[in] device_id The ID of the device to perform operations.
   *
   */
  Flat(int d, cnindexMetric_t metric, int device_id);

  /**
   * @brief Destructor.
   *
   */
  ~Flat();

  /**
   * @brief Deleted copy constructor.
   */
  Flat(const Flat&) = delete;

  /**
   * @brief Deleted copy overloading assign operator.
   */
  Flat &operator=(const Flat&) = delete;

  /**
   * @brief Move constructor.
   */
  Flat(Flat&&);

  /**
   * @brief Move overloading assign operator.
   */
  Flat &operator=(Flat&&);

  /**
   * @brief Resets a flat index. This function clears all vectors in the flat index.
   *
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   *
   */
  cnindexReturn_t Reset();

  /**
   * @brief Searchs vectors in a flat index.
   *
   * @param[in] n The number of vectors to search.
   * @param[in] x Pointer to the host memory that stores the input vectors to search.
   *              The size of vectors equals \b n * \b d , where \b d is the dimensionality of vector.
   * @param[in] k The TopK for searching.
   * @param[out] ids Pointer to the host memory that stores the IDs of the searching results.
   *             The size of IDs equals \b n * \b k .
   * @param[out] distances Pointer to the host memory that stores the distances of the searching results.
   *             The size of distances equals \b n * \b k .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   * 
   */
  cnindexReturn_t Search(int n, const float *x, int k, int *ids, float *distances = nullptr) const;

  /**
   * @brief Adds vectors to a flat index.
   *
   * @param[in] n The number of vectors to add.
   * @param[in] x Pointer to the host memory that stores the vectors to add.
   *              The size of vectors equals \b n * \b d , where \b d is the dimensionality of vector.
   * @param[in] ids Pointer to the host memory that stores the IDs of the adding vectors.
   *                The size of IDs equals \b n .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   * 
   */
  cnindexReturn_t Add(int n, const float *x, const int *ids = nullptr);

  /**
   * @brief Removes vectors from a flat index.
   *
   * @param[in] n The number of vectors to remove.
   * @param[in] ids Pointer to the host memory that stores the IDs of the removing vectors.
   *                The size of IDs equals \b n .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   *
   */
  cnindexReturn_t Remove(int n, const int *ids);

  /**
   * @brief Gets the dimensionality of vectors in a flat index.
   *
   * @retval 
   * - >0 The dimensionality of vectors.
   * - <0 The error code is returned.
   */
  int GetDimension() const;

  /**
   * @brief Gets the number of vectors in a flat index.
   *
   * @retval 
   * - >=0 The number of vectors.
   * - <0 The error code is returned.
   */
  int GetSize() const;

  /**
   * @brief Gets vectors from a flat index.
   *
   * @param[out] x Pointer to the host memory that stores the vectors.
   *               The size of vectors equals \b ntotal * \b d ,
   *               where \b ntotal is the total number of vectors in a flat index
   *               and \b d is the dimensionality of vectors.
   * @param[out] ids Pointer to the host memory that stores the IDs of vectors.
   *                 The size of IDs equals \b ntotal .
   * @return
   * - Returns ::CNINDEX_RET_SUCCESS if success. Otherwise returns a code error.
   *   For details on error codes, see ::cnindexReturn_t .
   *
   */
  cnindexReturn_t GetData(float *x, int *ids = nullptr) const;

 private:
  impl::Flat *flat_ = nullptr;

  friend impl::IVFPQ2;
  friend impl::IVFPQ3;
};  // Flat

}  // cnindex

#endif  // __CNINDEX_FLAT_H__
