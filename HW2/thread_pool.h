#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <cstdlib>
#include <functional>

namespace internal
{
    void enqueue(std::function<void(void)>&& f);

    bool done();
}

template<typename F>
void parallel_for_loop(size_t begin_idx, size_t end_idx, F&& f)
{
    for(size_t i = begin_idx; i < end_idx; ++i)
        internal::enqueue([&f, i](){ f(i); });
    while(!internal::done());
}

#endif
