#pragma once

#include "NumCpp/NdArray.hpp"


template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step);

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop);

template <typename IntType>
std::vector<IntType> range(IntType stop);

template <typename T>
std::string vectorToString(std::vector<T> v);

template <typename T>
std::string vectorToString(nc::NdArray<T> v);

template <typename T>
std::string matrixToString(std::vector<std::vector<T> > m, int numRows = 0, int numCols = 0);

template <typename T>
std::string matrixToString(nc::NdArray<T> m, int numRows = 0, int numCols = 0);

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1);

template<typename T>
std::vector<T> arange(T stop);

template<typename T>
std::vector<T> linspace(T start_in, T end_in, int num_in, bool endpoint = true);

template <typename T> 
int sgn(T val);