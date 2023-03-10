/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
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

#ifndef __CNINDEX_THREADSAFE_QUEUE_H__
#define __CNINDEX_THREADSAFE_QUEUE_H__

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace cnindex {

/**
 * @brief Thread-safe queue
 *
 * @tparam T Type of stored elements
 * @tparam Q Type of underlying container to store the elements, which acts as queue,
 *           `std::queue` and `std::priority_queue` satisfy the requirements
 */
template <typename T, typename Container = std::queue<T>>
class ThreadSafeQueue {
 public:
  /// type of container
  using queue_type = typename std::enable_if<std::is_same<typename Container::value_type, T>::value, Container>::type;
  /// type of elements
  using value_type = T;
  /// Container::size_type
  using size_type = typename Container::size_type;

  /**
   * @brief Construct a new Thread Safe Queue object
   */
  ThreadSafeQueue() = default;

  /**
   * @brief Try to pop an element from queue
   *
   * @param value An element
   * @retval true Succeed
   * @retval false Fail, no element stored in queue
   */
  bool TryPop(T& value);

  /**
   * @brief Try to pop an element from queue, wait for `timeout` if queue is empty
   *
   * @param value An element
   * @param timeout_ms Maximum duration in unit of milliseconds to block for,
   *                <0: means wait forever; =0: same with TryPop; >0: timeout in milliseconds
   * @retval true Succeed
   * @retval false Timeout
   */
  bool WaitAndTryPop(T& value, int timeout_ms);

  /**
   * @brief Pushes the given element value to the end of the queue
   *
   * @param new_value the value of the element to push
   */
  void Push(const T& new_value) {
    std::lock_guard<std::mutex> lk(mtx_);
    q_.push(new_value);
    cv_.notify_one();
  }

  /**
   * @brief Pushes the given element value to the end of the queue
   *
   * @param new_value the value of the element to push
   */
  void Push(T&& new_value) {
    std::lock_guard<std::mutex> lk(mtx_);
    q_.push(std::move(new_value));
    cv_.notify_one();
  }

  /**
   * @brief Pushes a new element to the end of the queue. The element is constructed in-place.
   *
   * @tparam Arguments Type of arguments to forward to the constructor of the element
   * @param args Arguments to forward to the constructor of the element
   */
  template <typename... Arguments>
  void Emplace(Arguments&&... args) {
    std::lock_guard<std::mutex> lk(mtx_);
    q_.emplace(std::forward<Arguments>(args)...);
    cv_.notify_one();
  }

  /**
   * @brief Checks if the underlying container has no elements
   *
   * @retval true If the underlying container is empty
   * @retval false Otherwise
   */
  bool Empty() {
    std::lock_guard<std::mutex> lk(mtx_);
    return q_.empty();
  }

  /**
   * @brief Returns the number of elements in the underlying container
   *
   * @return size_type The number of elements in the container
   */
  size_type Size() {
    std::lock_guard<std::mutex> lk(mtx_);
    return q_.size();
  }

 private:
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;

  std::mutex mtx_;
  queue_type q_;
  std::condition_variable cv_;
};  // class ThreadSafeQueue

namespace detail {
template <typename T, typename = typename std::enable_if<!std::is_move_assignable<T>::value>::type>
inline void GetFrontAndPop(std::queue<T>* q_, T* value) {
  *value = q_->front();
  q_->pop();
}

template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>,
          typename = typename std::enable_if<!std::is_move_assignable<T>::value>::type>
inline void GetFrontAndPop(std::priority_queue<T, Container, Compare>* q_, T* value) {
  *value = q_->top();
  q_->pop();
}

template <typename T>
inline void GetFrontAndPop(std::queue<T>* q_, T* value) {
  *value = std::move(q_->front());
  q_->pop();
}

template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
inline void GetFrontAndPop(std::priority_queue<T, Container, Compare>* q_, T* value) {
  // cut off const to enable move
  *value = std::move(const_cast<T&>(q_->top()));
  q_->pop();
}
}  // namespace detail

template <typename T, typename Q>
bool ThreadSafeQueue<T, Q>::TryPop(T& value) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (q_.empty()) {
    return false;
  }

  detail::GetFrontAndPop<T>(&q_, &value);
  return true;
}

template <typename T, typename Q>
bool ThreadSafeQueue<T, Q>::WaitAndTryPop(T& value, int timeout_ms) {
  if (timeout_ms == 0) return TryPop(value);
  std::unique_lock<std::mutex> lk(mtx_);
  if (timeout_ms < 0) {
    cv_.wait(lk, [&] { return !q_.empty(); });
    detail::GetFrontAndPop<T>(&q_, &value);
    return true;
  } else {
    if (cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !q_.empty(); })) {
      detail::GetFrontAndPop<T>(&q_, &value);
      return true;
    } else {
      return false;
    }
  }
}

/**
 * @brief Alias of ThreadSafeQueue<T, std::queue<T>>
 *
 * @tparam T Type of stored elements
 */
template <typename T>
using TSQueue = ThreadSafeQueue<T, std::queue<T>>;

/**
 * @brief Alias of ThreadSafeQueue<T, std::priority_queue<T>>
 *
 * @tparam T Type of stored elements
 */
template <typename T>
using TSPriorityQueue = ThreadSafeQueue<T, std::priority_queue<T>>;

}  // namespace cnindex

#endif  // __CNINDEX_THREADSAFE_QUEUE_H__
