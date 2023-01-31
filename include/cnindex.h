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

#ifndef __CNINDEX_H__
#define __CNINDEX_H__

/*!
 * @enum cnindexReturn_t
 *
 * @brief Enumeration variables describing the return values of CNIndex API calls.
 *
 */
typedef enum {
  CNINDEX_RET_SUCCESS         = 0,   /*!< Successfully run the function.*/
  CNINDEX_RET_NOT_IMPL        = -1,  /*!< Operation not supported. */
  CNINDEX_RET_NOT_VALID       = -2,  /*!< Invalid handle status.*/
  CNINDEX_RET_BAD_PARAMS      = -3,  /*!< Illegal parameters. */
  CNINDEX_RET_ALLOC_FAILED    = -4,  /*!< Failed to allocate memory. */
  CNINDEX_RET_OP_FAILED       = -5,  /*!< Failed to execute the MLU operator.*/
} cnindexReturn_t;

/*!
 * @enum cnindexMetric_t
 *
 * @brief Enumeration variables describing the metric type of distance computation.
 *
 */
typedef enum {
  CNINDEX_METRIC_L1           = 0,   /*!< Manhattan distance. */
  CNINDEX_METRIC_L2           = 0,   /*!< Euclidean distance.*/
  CNINDEX_METRIC_IP           = 1,   /*!< Inner product. */
} cnindexMetric_t;

#include "cnindex_flat.h"
#include "cnindex_ivfpq.h"
#include "cnindex_pq.h"

#endif  // __CNINDEX_H__
