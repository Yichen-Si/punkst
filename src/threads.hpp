#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// Thread-safe queue
template <typename T>
class ThreadSafeQueue {
public:
    void push(const T& value) {

        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(value);
        cv_.notify_one();
    }

    // Pop a value if available. Returns false if finished and queue is empty.
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (!queue_.empty()) {
            value = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(mtx_);
        done_ = true;
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool done_ = false;
};
