#include <string>
#include <sstream> 
#include <vector> 


template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step)
{
    if (step == IntType(0))
    {
        throw std::invalid_argument("step for range must be non-zero");
    }

    std::vector<IntType> result;
    IntType i = start;
    while ((step > 0) ? (i < stop) : (i > stop))
    {
        result.push_back(i);
        i += step;
    }
    return result;
}


template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop)
{
    return range(start, stop, IntType(1));
}


template <typename IntType>
std::vector<IntType> range(IntType stop)
{
    return range(IntType(0), stop, IntType(1));
}


template <typename T>
std::string vectorToString(std::vector<T> v)
{
    std::stringstream ss; 
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            ss << ", ";
        ss << v[i];
    }
    std::string s = ss.str();
    return s;
}


template std::vector<float> range(float start, float stop);
template std::vector<int> range(int start, int stop);
template std::string vectorToString(std::vector<float>);
