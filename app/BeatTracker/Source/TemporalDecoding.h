#ifndef TEMPORAL_DECODING_H
#define TEMPORAL_DECODING_H


#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
#include <cfloat>
#include <vector> 

#include "NumCpp/Functions.hpp"
#include "NumCpp/NdArray.hpp"
#include "utils.h"


class StateSpace
{
public:
	StateSpace();

	~StateSpace();

	int numStates;
	nc::NdArray<int> intervals;
	nc::NdArray<int> firstStates;
	nc::NdArray<int> lastStates;
	nc::NdArray<double> statePositions;
};


class TransitionModel
{
public:
	TransitionModel(StateSpace stateSpace);

	~TransitionModel();

	nc::NdArray<double> exponentialTransition(nc::NdArray<int> &fromInt, 
		nc::NdArray<int> &toInt, double &transitionLambda);

	int numStates;
	nc::NdArray<nc::uint32> tmIndices;
	nc::NdArray<nc::uint32> tmPointers;
	nc::NdArray<double> tmProbabilities;
	nc::NdArray<double> initialDistribution;
};


class ObservationModel
{
public:
	ObservationModel(StateSpace &stateSpace, nc::NdArray<double> &observations);

	~ObservationModel();

	nc::NdArray<nc::uint32> omPointers;
	nc::NdArray<double> omDensities;
};


nc::NdArray<nc::uint32> viterbi(nc::NdArray<double> &observations, TransitionModel &tm, ObservationModel &om);


std::vector<double> activationsToBeats(std::vector<double> &activations);


#endif