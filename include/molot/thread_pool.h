#pragma once
// Persistent thread pool: 1 worker per core (pinned), caller also computes.
// Static partition: participant p in [0,P) owns [p*chunk,(p+1)*chunk).
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace molot {

class ThreadPool {
 public:
  explicit ThreadPool(int nthreads = 0) {
    if (nthreads <= 0) {
      nthreads = (int)std::thread::hardware_concurrency();
      if (nthreads <= 0) nthreads = 4;
    }
    stop_ = false;
    gen_ = 0;
    done_ = 0;
    for (int i = 0; i < nthreads - 1; ++i) {
      workers_.emplace_back([this, i] { loop(i); });
#ifdef __linux__
      cpu_set_t set;
      CPU_ZERO(&set);
      int ncores = (int)std::thread::hardware_concurrency();
      CPU_SET(ncores > 0 ? (i + 1) % ncores : 0, &set);
      pthread_setaffinity_np(workers_.back().native_handle(), sizeof(set), &set);
#endif
    }
  }
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lk(mu_);
      stop_ = true;
      cv_.notify_all();
    }
    for (auto& t : workers_)
      if (t.joinable()) t.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return (int)workers_.size() + 1; }

  template <typename Fn>
  void parallel_for(int64_t n, Fn&& fn, int64_t min_chunk = 1) {
    if (n <= 0) return;
    int P = size();
    if (P == 1 || n <= min_chunk) {
      for (int64_t i = 0; i < n; ++i) fn(P - 1, i);
      return;
    }
    int64_t chunk = (n + P - 1) / P;
    {
      std::unique_lock<std::mutex> lk(mu_);
      fn_ = [&](int tid) {
        int64_t s = (int64_t)tid * chunk;
        int64_t e = s + chunk < n ? s + chunk : n;
        if (s < n)
          for (int64_t i = s; i < e; ++i) fn(tid, i);
      };
      done_ = 0;
      my_gen_ = ++gen_;
      cv_.notify_all();
    }
    // caller is participant P-1
    {
      int tid = P - 1;
      int64_t s = (int64_t)tid * chunk;
      int64_t e = s + chunk < n ? s + chunk : n;
      if (s < n)
        for (int64_t i = s; i < e; ++i) fn(tid, i);
    }
    // wait for workers
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return done_ == (int)workers_.size() || stop_; });
      fn_ = nullptr;
      cv_.notify_all();
    }
  }

 private:
  void loop(int tid) {
    uint64_t seen = 0;
    for (;;) {
      std::function<void(int)> f;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        f = fn_;
      }
      if (f) f(tid);
      {
        std::unique_lock<std::mutex> lk(mu_);
        // only count if this generation still current (avoid double count)
        if (seen == gen_) {
          if (++done_ == (int)workers_.size()) cv_.notify_all();
          // wait until main consumes (fn_ cleared) or next gen or stop.
          // Do NOT resync seen here: outer wait needs gen_ != seen to pick
          // up a new generation that arrived while we were finishing.
          cv_.wait(lk, [&] { return stop_ || fn_ == nullptr || gen_ != seen; });
          if (stop_) return;
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::function<void(int)> fn_{nullptr};
  uint64_t gen_{0}, my_gen_{0};
  int done_{0};
  bool stop_{false};
};

}  // namespace molot
