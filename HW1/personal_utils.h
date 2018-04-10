#ifndef _PERSONAL_UTILS_H_
#define _PERSONAL_UTILS_H_

#include <random>
#include <chrono>

template<typename It, typename Type>
inline void fill_random(It begin, It end, Type min, Type max)
{
    auto dist = std::uniform_real_distribution<Type>(min, max);
    auto rd = std::mt19937(std::random_device{}());
    for (It i = begin; i != end; ++i) {
        *i = dist(rd);
    }
}

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

template<typename Obj, typename Func>
inline double benchmark(Obj&& obj, Func fun, bool verbose = false)
{
    auto start = std::chrono::steady_clock::now();
    (obj.*fun)();
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    if(verbose)
    fprintf(stdout, "     * Time by host clock = %.3fus\n", duration);
    return duration;
}

template<typename It, typename Type>
inline void fill(It begin, It end, Type value)
{
    for(It i = begin; i != end; ++i) 
        *i = value;
}

template<typename It>
inline It max(It begin, It end)
{
    It result = begin;
    for(It i = begin; i != end; ++i) 
    {
        if(*result < *i)
            result = i;
    }
    return result;
}

template<typename It>
inline It min(It begin, It end)
{
    It result = begin;
    for(It i = begin; i != end; ++i) 
    {
        if(*result > *i)
            result = i;
    }
    return result;
}

template<typename It>
inline double avg(It begin, It end)
{
    size_t n = 1;
    double sum = 0;
    for(It i = begin; i != end; ++i) 
    {
        sum += *i;
        ++n;
    }
    return sum / n;
}

template<typename It>
inline double var(It begin, It end, double avg)
{
    size_t n = 0;
    double sum = 0;
    for(It i = begin; i != end; ++i) 
    {
        ++n;
        double temp = (*i) - avg;
        sum += temp * temp;
    }
    return sum / n;
}


inline size_t read_file(char const* filename,
                        char** src,
                        bool verbose = false)
{
    FILE *fp;
    size_t count;
    
    if ((fp = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Error: cannot open the file %s for reading...\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fseek(fp, 0, SEEK_END);
    count = ftell(fp);
    
    fseek(fp, 0, SEEK_SET);
    
    *src = (char*)malloc(count + 1);
    if (*src == NULL) {
        fprintf(stderr, "Error: cannot allocate memory for reading the file %s for reading...\n", filename);
    }
    
    fread(*src, sizeof(char), count, fp);
    fclose(fp);

    (*src)[count] = '\0';
    
    if(verbose)
        printf("\n^^^^^^^^^^^^^^ The OpenCL C program ^^^^^^^^^^^^^^\n%s\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n", *src);
    
    return count;
}

template<typename It, typename Type>
inline Type kahan_reduce(It begin, It end, Type init)
{
    Type sum = init;
    Type running_error = Type();
    Type temp;
    Type difference;

    for (; begin != end; ++begin) {
        difference = *begin;
        difference -= running_error;
        temp = sum;
        temp += difference;
        running_error = temp;
        running_error -= sum;
        running_error -= difference;
        sum = temp;
    }
    return sum;
}

template<class T, class U>
inline void eval(T&& object, U fun)
{
    (object.*fun)();
}


template<typename Suite>
void run_benchmark(Suite&& benchmark_suite,
                   size_t* group_dim,
                   size_t* problem_size,
                   size_t dims)
{
    benchmark_suite.init(problem_size);

//cl_event event_for_timing;

    // size_t local_dim[Dim];
    benchmark_suite.work_dimensions(group_dim);

    size_t const sample_num = 200;
    double gpu_runtime[sample_num];
    double cpu_runtime[sample_num];
    for(size_t i = 0; i < sample_num; ++i)
    {
        benchmark_suite.prepare();
        gpu_runtime[i] = benchmark(benchmark_suite, &Suite::run_gpu);
        cpu_runtime[i] = benchmark(benchmark_suite, &Suite::run_cpu);

        if(!benchmark_suite.check_result())
            exit(2);

        benchmark_suite.teardown();
    }

    double gpu_average = avg(gpu_runtime, gpu_runtime + sample_num);
    double gpu_variance = var(gpu_runtime, gpu_runtime + sample_num, gpu_average);
    double gpu_stddev = sqrt(gpu_variance);

    double cpu_average = avg(cpu_runtime, cpu_runtime + sample_num);
    double cpu_variance = var(cpu_runtime, cpu_runtime + sample_num, cpu_average);
    double cpu_stddev = sqrt(cpu_variance);

    size_t total_size = 1;
    for(size_t i = 0; i < dims; ++i)
        total_size *= problem_size[0];

    printf(" problem size: %8d, [gpu] mean: %5.3fus (stddev %.3fus), [cpu] mean: %5.3fus (stddev %.3fus)\n",
           (int)total_size, gpu_average, gpu_stddev, cpu_average, cpu_stddev);
}

#endif
