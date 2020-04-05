#include <string>
#include <sstream> 
#include <vector> 

#include "NumCpp/NdArray.hpp"


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


template <typename T>
std::string vectorToString(nc::NdArray<T> v)
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



template <typename T>
std::string matrixToString(std::vector<std::vector<T> > m, int numRows = 0, int numCols = 0)
{
    if (numRows == 0)
        numRows = m.size();

    std::stringstream ss; 
    for(int i = 0 ; i < numRows ; ++i)
    {
        if (numCols == 0)
            numCols = m[i].size();
        for(int j = 0 ; j < numCols ; ++j)
        {
            if(j != 0)
            {
                ss << ", ";
            }
            else
            {
                ss << "[";
            }
            ss << m.at(i).at(j);
        }
        ss << "]\n";
    }
    std::string s = ss.str();
    return s;
}


template <typename T>
std::string matrixToString(nc::NdArray<T> m, int numRows = 0, int numCols = 0)
{
    if (numRows == 0)
        numRows = m.shape().rows;

    std::stringstream ss; 
    for(int i = 0 ; i < numRows ; ++i)
    {
        if (numCols == 0)
            numCols = m.shape().cols;
        for(int j = 0 ; j < numCols ; ++j)
        {
            if(j != 0)
            {
                ss << ", ";
            }
            else
            {
                ss << "[";
            }
            ss << m(i, j);
        }
        ss << "]\n";
    }
    std::string s = ss.str();
    return s;
}


template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) 
{
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}


template<typename T>
std::vector<T> arange(T stop) 
{
    std::vector<T> values;
    for (T value = 0; value < stop; ++value)
        values.push_back(value);
    return values;
}


template<typename T>
std::vector<T> linspace(T startIn, T endIn, int numIn, bool endpoint = true)
{
    std::vector<T> linspaced;

    T start = static_cast<T>(startIn);
    T end = static_cast<T>(endIn);
    T num = static_cast<T>(numIn);

    if (num == 0) 
    { 
        return linspaced; 
    }
    if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

    if (endpoint)
    {
        T delta = (end - start) / (num - 1);
        for(int i=0; i < num-1; ++i)
        {
            linspaced.push_back(start + delta * i);
        }
        linspaced.push_back(end); 
    }
    else
    {
        T delta = (end - start) / num;
        for(int i=0; i < num; ++i)
        {
            linspaced.push_back(start + delta * i);
        }
    }
    return linspaced;
}


template<typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}



template std::vector<float> range(float start, float stop);
template std::vector<int> range(int start, int stop);
template std::string vectorToString(std::vector<float>);
template std::string vectorToString(std::vector<double>);
template std::string vectorToString(std::vector<int>);
template std::string vectorToString(std::vector<unsigned int>);
template std::string vectorToString(nc::NdArray<nc::uint32>);
template std::string vectorToString(nc::NdArray<float>);
template std::string vectorToString(nc::NdArray<double>);
template std::string matrixToString(nc::NdArray<float> m, int numRows = 0, int numCols = 0);
template std::string matrixToString(nc::NdArray<double> m, int numRows = 0, int numCols = 0);
template std::string matrixToString(nc::NdArray<nc::uint32> m, int numRows = 0, int numCols = 0);
template std::vector<int> arange(int start, int stop, int step = 1);
template std::vector<int> arange(int stop);
template std::vector<float> linspace(float start, float stop, int numIn, bool endpoint = true);
template int sgn(double val);

