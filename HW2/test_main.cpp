
#include <iostream>
#include <chrono>

#include "thread_pool.h"

#define SIZE 100000
int work[SIZE] = {0};

template<typename Func>
inline double benchmark(Func&& fun, bool verbose = false)
{
    auto start = std::chrono::steady_clock::now();
    fun();
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    if(verbose)
        fprintf(stdout, "     * Time by host clock = %.3fus\n", duration);
    return duration;
}

int main()
{
    benchmark([&](){
            for(auto i = 0u; i < SIZE; ++i)
            {
                work[i] = 1;
            }
        }, true);

    benchmark([&](){
            parallel_for_loop(0, SIZE, [&](size_t i){
                    work[i] = 1;
                });
        }, true);

    for(auto i = 0u; i < SIZE; ++i)
    {
        if(work[i] != 1)
        {
            std::cout << "somethings wrong! \n" << std::endl; 
        }
                
    }
    std::cout << "done!" << std::endl;
    return 0;
}
