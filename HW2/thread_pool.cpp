
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>
#include <algorithm>
#include <condition_variable>

#include "thread_pool.h"
#include "parameters.h"

struct task_queue
{
    std::mutex _mtx;
    std::condition_variable _cond;
    std::queue<std::function<void(void)>> _queue;
    std::atomic<bool> _waiting;
};

void thread_task(task_queue* queue)
{
    while(true)
    {
        std::unique_lock<std::mutex> _lck(queue->_mtx);
        {
            queue->_waiting.store(true);
            queue->_cond.wait(_lck, [&](){ return !queue->_queue.empty(); });
            queue->_waiting.store(false);
        }
        auto f = std::move(queue->_queue.front());
        queue->_queue.pop();

        f();
    } 
}

class thread_pool
 {
     size_t _threads;
     std::vector<std::thread> _workers;
     std::vector<task_queue> _tasks;
     size_t mux;

 public:
     thread_pool()
         :_threads(
             std::min(std::thread::hardware_concurrency() - 1,
                      static_cast<unsigned>(MAX_THREADS))),
          _workers(),
          _tasks(_threads),
          mux(0)
     {
         _workers.reserve(_threads);
         for(size_t i = 0; i < _threads; ++i)
         {
             _workers.emplace_back(thread_task, &_tasks[i]);
             _workers.back().detach();
         }
     }

     void enqueue(std::function<void(void)>&& f)
     {
         if(mux >= _tasks.size())
             mux = 0;

         while(!_tasks[mux]._mtx.try_lock())
         {
             ++mux;
             if(mux== _tasks.size())
                 mux = 0;
         }
         _tasks[mux]._queue.push(std::move(f));
         _tasks[mux]._mtx.unlock();
         _tasks[mux]._cond.notify_one();
         ++mux;
     } 

     bool done()
     {
         if(std::all_of(_tasks.begin(), _tasks.end(),
                        [](task_queue const& queue){
                                return queue._queue.empty();
                        }))
         {
             
             return std::all_of(_tasks.begin(), _tasks.end(),
                                [](task_queue const& queue){
                                    return queue._waiting.load();
                                });
         }
         else
             return false;
     }

} _pool;

bool
internal::
done()
{
    return _pool.done();
}
    
void
internal::
enqueue(std::function<void(void)>&& f)
{
    _pool.enqueue(std::move(f));
}
