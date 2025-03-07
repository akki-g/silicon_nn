#ifndef PARALLEL_H
#define PARALLEL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <algorithm> // for std::min

class ThreadPool {
public: 
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; i++) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);   
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    size_t getWorkerCount() const {
        return workers.size();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

template <typename IndexType, typename Function>
void parallel_for(IndexType start, IndexType end, Function f, ThreadPool &pool) {
    IndexType total = end - start;
    // Use a minimum block size to reduce scheduling overhead.
    const IndexType minBlockSize = 64; // Tune this parameter as needed.
    
    // If the total work is too small, run serially.
    if (total < minBlockSize) {
        for (IndexType i = start; i < end; i++) {
            f(i);
        }
        return;
    }

    unsigned int numThreads = pool.getWorkerCount();
    IndexType blockSize = total / numThreads;
    // Ensure each block does at least minBlockSize work.
    if (blockSize < minBlockSize) blockSize = minBlockSize;
    IndexType totalBlocks = (total + blockSize - 1) / blockSize;

    std::vector<std::future<void>> futures;
    for (IndexType block = 0; block < totalBlocks; block++) {
        IndexType blockStart = start + block * blockSize;
        IndexType blockEnd = std::min(end, blockStart + blockSize);
        futures.emplace_back(pool.enqueue([=]() {
            for (IndexType j = blockStart; j < blockEnd; j++) {
                f(j);
            }
        }));
    }
    for (auto &fut : futures) {
        fut.get();
    }
}

#endif // PARALLEL_H
