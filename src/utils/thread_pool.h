/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * A part of this source code is referenced from ctpl project.
 * https://github.com/vit-vit/CTPL/blob/master/ctpl_stl.h
 * Copyright (C) 2014 by Vitaliy Vitsentiy
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

#ifndef __CNINDEX_THREAD_POOL_H__
#define __CNINDEX_THREAD_POOL_H__

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "log.h"
#include "threadsafe_queue.h"

namespace cnindex {

/**
 * @brief Task functor
 */
struct Task {
  /// Invoke the task function
  void operator()() {
    if (func) {
      (func)();
    } else {
      LOGW(ThreadPool) << "No task function";
    }
  }

  /// Function to be invoked
  std::function<void()> func = nullptr;
  /// Task priority
  int64_t priority = 0;
  /**
   * @brief Construct a new Task object
   */
  Task() = default;
  /**
   * @brief Construct a new Task object
   *
   * @param f Function to be invoked
   * @param p Task priority
   */
  Task(std::function<void()>&& f, int64_t p) : func(std::forward<std::function<void()>>(f)), priority(p) {}

  /**
   * @brief Function object for performing comparisons between tasks
   */
  struct Compare {
    /**
     * @brief Checks whether priority of the first task is less than the second
     *
     * @param lhs One task
     * @param rhs Another task
     * @retval true If lhs.priority < rhs.priority
     * @retval false Otherwise
     */
    bool operator()(const Task &lhs, const Task &rhs) { return lhs.priority < rhs.priority; }
  };
};

/**
 * @brief Thread pool to run user's functors with signature `ret func(params)`
 *
 * @tparam Q Type of underlying container to store the tasks
 */
template <typename Q = TSQueue<Task>, typename = std::enable_if<std::is_same<typename Q::value_type, Task>::value>>
class ThreadPool {
 public:
  /// Type of container
  using queue_type = Q;
  /// Type of task
  using task_type = typename std::enable_if<std::is_same<typename Q::value_type, Task>::value, Task>::type;

  /**
   * @brief Construct a new Thread Pool object
   *
   * @param th_init_func Init function invoked at start of each thread in pool
   * @param n_threads Number of threads
   */
  explicit ThreadPool(std::function<bool()> th_init_func, int n_threads = 0) : thread_init_func_(th_init_func) {
    if (n_threads > 0) Resize(n_threads);
  }

  /**
   * @brief Destroy the Thread Pool object
   *
   * @note the destructor waits for all the functions in the queue to be finished
   */
  ~ThreadPool() { Stop(true); }

  /**
   * @brief Get the number of threads in the pool
   *
   * @return size_t Number of threads
   */
  size_t Size() const noexcept { return threads_.size(); }

  /**
   * @brief Get the number of idle threads in the pool
   *
   * @return int Number of idle threads
   */
  uint32_t IdleNumber() const noexcept { return n_idle_.load(); }

  /**
   * @brief Get the Thread at the specified index
   *
   * @param i The specified index
   * @return std::thread& A thread
   */
  std::thread &GetThread(int i) { return *threads_[i]; }

  /**
   * @brief Change the number of threads in the pool
   *
   * @warning Should be called from one thread, otherwise be careful to not interleave, also with this->stop()
   * @param n_threads Target number of threads
   */
  void Resize(size_t n_threads) noexcept;

  /**
   * @brief Wait for all computing threads to finish and stop all threads
   *
   * @param wait_all_task_done If wait_all_task_done == true, all the functions in the queue are run,
   *                           otherwise the queue is cleared without running the functions
   */
  void Stop(bool wait_all_task_done = false) noexcept;

  /**
   * @brief Empty the underlying queue
   */
  void ClearQueue() {
    task_type t;
    // empty the queue
    while (task_q_.TryPop(t)) {
    }
  }

  /**
   * @brief Pops a task
   *
   * @return task_type A task
   */
  task_type Pop() {
    task_type t;
    task_q_.TryPop(t);
    return t;
  }

  /**
   * @brief Run the user's function, returned value is templatized in future
   *
   * @tparam callable Type of callable object
   * @tparam arguments Type of arguments passed to callable
   * @param priority Task priority
   * @param f Callable object to be invoked
   * @param args Arguments passed to callable
   * @return std::future<typename std::result_of<callable(arguments...)>::type>
   *         A future that wraps the returned value of user's function,
   *         where the user can get the result and rethrow the catched exceptions
   */
  template <typename callable, typename... arguments>
  auto Push(int64_t priority, callable &&f, arguments &&... args)
      -> std::future<typename std::result_of<callable(arguments...)>::type> {
    LOGD(ThreadPool) << "Sumbit one task to threadpool, priority: " << priority;
    LOGD(ThreadPool) << "thread pool (idle/total): " << IdleNumber() << " / " << Size();
    auto pck = std::make_shared<std::packaged_task<typename std::result_of<callable(arguments...)>::type()>>(
        std::bind(std::forward<callable>(f), std::forward<arguments>(args)...));
    task_q_.Emplace([pck]() { (*pck)(); }, priority);
    cv_.notify_one();
    return pck->get_future();
  }

  /**
   * @brief Run the user's function with default priority 0, returned value is templatized in future
   *
   * @tparam callable Type of callable object
   * @tparam arguments Type of arguments passed to callable
   * @param f Callable object to be invoked
   * @param args Arguments passed to callable
   * @return std::future<typename std::result_of<callable(arguments...)>::type>
   *         A future that wraps the returned value of user's function,
   *         where the user can get the result and rethrow the catched exceptions
   */
  template <typename callable, typename... arguments>
  auto Push(callable &&f, arguments &&... args)
      -> std::future<typename std::result_of<callable(arguments...)>::type> {
    return Push(0, std::forward<callable>(f), std::forward<arguments>(args)...);
  }

  /**
   * @brief Run the user's function without returned value
   *
   * @warning There's no future to wrap exceptions, therefore user should guarantee that task won't throw,
   *          otherwise the program may be corrupted
   * @tparam callable Type of callable object
   * @tparam arguments Type of arguments passed to callable
   * @param priority Task priority
   * @param f Callable object to be invoked
   * @param args Arguments passed to callable
   */
  template <typename callable, typename... arguments>
  void VoidPush(int64_t priority, callable &&f, arguments &&... args) {
    LOGD(ThreadPool) << "Sumbit one task to threadpool, priority: " << priority;
    LOGD(ThreadPool) << "thread pool (idle/total): " << IdleNumber() << " / " << Size();
    task_q_.Emplace(std::bind(std::forward<callable>(f), std::forward<arguments>(args)...), priority);
    cv_.notify_one();
  }

  /**
   * @brief Run the user's function with default priority 0 without returned value
   *
   * @warning There's no future to wrap exceptions, therefore user should guarantee that task won't throw,
   *          otherwise the program may be corrupted
   * @tparam callable Type of callable object
   * @tparam arguments Type of arguments passed to callable
   * @param f Callable object to be invoked
   * @param args Arguments passed to callable
   */
  template <typename callable, typename... arguments>
  void VoidPush(callable &&f, arguments &&... args) {
    VoidPush(0, std::forward<callable>(f), std::forward<arguments>(args)...);
  }

 private:
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  void SetThread(int i) noexcept;

  std::vector<std::unique_ptr<std::thread>> threads_;
  std::vector<std::shared_ptr<std::atomic<bool>>> flags_;
  queue_type task_q_;
  std::atomic<bool> is_done_{false};
  std::atomic<bool> is_stop_{false};
  // how many threads are idle
  std::atomic<uint32_t> n_idle_{0};

  std::mutex mutex_;
  std::condition_variable cv_;

  std::function<bool()> thread_init_func_{nullptr};
};  // class ThreadPool

/// Alias of ThreadPool<TSQueue<Task>>
using EqualityThreadPool = ThreadPool<TSQueue<Task>>;
/// Alias of ThreadPool<ThreadSafeQueue<Task, std::priority_queue<Task, std::vector<Task>, Task::Compare>>>
using PriorityThreadPool =
    ThreadPool<ThreadSafeQueue<Task, std::priority_queue<Task, std::vector<Task>, Task::Compare>>>;

/* ----------------- Implement --------------------- */
template <typename Q, typename T>
void ThreadPool<Q, T>::Resize(size_t n_threads) noexcept {
  if (!is_stop_ && !is_done_) {
    size_t old_n_threads = threads_.size();
    if (old_n_threads <= n_threads) {
      // if the number of threads is increased
      LOGD(ThreadPool) << "add " << n_threads - old_n_threads << " threads into thread pool, total "
                       << n_threads << " threads";
      threads_.resize(n_threads);
      flags_.resize(n_threads);

      for (size_t i = old_n_threads; i < n_threads; ++i) {
        flags_[i] = std::make_shared<std::atomic<bool>>(false);
        SetThread(i);
      }
    } else {
      // the number of threads is decreased
      LOGD(ThreadPool) << "remove " << old_n_threads - n_threads << " threads in threadpool, remain "
                       << n_threads << " threads";
      for (size_t i = n_threads; i < old_n_threads; ++i) {
        // this thread will finish
        flags_[i]->store(true);
        threads_[i]->detach();
      }

      // stop the detached threads that were waiting
      cv_.notify_all();

      // safe to delete because the threads are detached
      threads_.resize(n_threads);
      // safe to delete because the threads have copies of shared_ptr of the flags, not originals
      flags_.resize(n_threads);
    }
  }
}

template <typename Q, typename T>
void ThreadPool<Q, T>::Stop(bool wait_all_task_done) noexcept {
  LOGD(ThreadPool) << "Before stop threadpool ----- Task number in queue: " << task_q_.Size()
                   << ", thread number: " << threads_.size() << ", idle number: " << IdleNumber();
  if (!wait_all_task_done) {
    if (is_stop_) return;
    LOGD(ThreadPool) << "stop all the thread without waiting for remained task done";
    is_stop_ = true;
    for (size_t i = 0, n = this->Size(); i < n; ++i) {
      // command the threads to stop
      flags_[i]->store(true);
    }

    // empty the queue
    this->ClearQueue();
  } else {
    if (is_done_ || is_stop_) return;
    LOGD(ThreadPool) << "waiting for remained task done before stop all the thread";
    // give the waiting threads a command to finish
    is_done_.store(true);
  }

  {
    // may stuck on thread::join if no lock here
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.notify_all();  // stop all waiting threads
  }

  // wait for the computing threads to finish
  for (size_t i = 0; i < threads_.size(); ++i) {
    if (threads_[i]->joinable()) threads_[i]->join();
  }

  // if there were no threads in the pool but some functors in the queue, the functors are not deleted by the threads
  // therefore delete them here
  this->ClearQueue();
  threads_.clear();
  flags_.clear();
}

template <typename Q, typename T>
void ThreadPool<Q, T>::SetThread(int i) noexcept {
  std::shared_ptr<std::atomic<bool>> tmp(flags_[i]);
  auto f = [this, i, tmp]() {
    std::atomic<bool> &flag = *tmp;
    // init params that bind with thread
    if (thread_init_func_) {
      if (thread_init_func_()) {
        LOGD(ThreadPool) << "Init thread context success, thread index: " << i;
      } else {
        LOGE(ThreadPool) << "Init thread context failed, but program will continue. "
                      "Program cannot work correctly maybe.";
      }
    }
    task_type t;
    bool have_task = task_q_.TryPop(t);
    while (true) {
      // if there is anything in the queue
      while (have_task) {
        t();
        // params encapsulated in std::function need destruct at once
        t.func = nullptr;
        if (flag.load()) {
          // the thread is wanted to stop, return even if the queue is not empty yet
          return;
        } else {
          have_task = task_q_.TryPop(t);
        }
      }

      // the queue is empty here, wait for the next command
      std::unique_lock<std::mutex> lock(mutex_);
      ++n_idle_;
      cv_.wait(lock, [this, &t, &have_task, &flag]() {
        have_task = task_q_.TryPop(t);
        return have_task || is_done_ || flag.load();
      });
      --n_idle_;

      // if the queue is empty and is_done_ == true or *flag then return
      if (!have_task) return;
    }
  };

  threads_[i].reset(new std::thread(f));
}
/* ----------------- Implement END --------------------- */

// instantiate thread pool
template class ThreadPool<TSQueue<Task>>;
template class ThreadPool<ThreadSafeQueue<Task, std::priority_queue<Task, std::vector<Task>, Task::Compare>>>;

}  // namespace cnindex

#endif  // __CNINDEX_THREAD_POOL_H__
