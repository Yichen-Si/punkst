# pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// Thread-safe queue
template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t cap = INT_MAX) : cap_(cap) {}
    void push(const T& value) {
        std::unique_lock lock(mtx_);
        cv_not_full_.wait(lock, [&]{ return queue_.size() < cap_ || done_; });
        if (done_) return;
        queue_.push(value);
        cv_not_empty_.notify_one();
    }
    void push(T&& value) { // rvalue-overload
        std::unique_lock lock(mtx_);
        cv_not_full_.wait(lock, [&]{ return queue_.size() < cap_ || done_; });
        if (done_) return;
        queue_.push(std::move(value));
        cv_not_empty_.notify_one();
    }

    // Pop a value if available. Returns false if finished and queue is empty.
    bool pop(T& value) {
        std::unique_lock lock(mtx_);
        cv_not_empty_.wait(lock, [&]{ return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();   // wake producer if it was blocked
        return true;
    }

    void set_done() {
        {
            std::lock_guard lock(mtx_);
            done_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    bool done_ = false;
    size_t cap_;
};
