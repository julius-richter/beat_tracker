#include "TemporalDecoding.h"


unsigned int fps = 100;
unsigned int minBPM = 56;
unsigned int maxBPM = 215;
double transitionLambda = 100.0;
double observationLambda = 16.0;
double threshold = 0.5;
double NP_SPACING_1 = 2.220446049250313e-16;


StateSpace::StateSpace()
{
	int minInterval = (int) (60.0 * (double) fps / (double) maxBPM);
	int maxInterval = (int) (60.0 * (double) fps / (double) minBPM);

	intervals = nc::arange(minInterval, maxInterval + 1);
	numStates = nc::sum(intervals)[0];
	firstStates = nc::cumsum(nc::hstack({{0}, intervals[nc::Slice(intervals.size()-1)]}));
	lastStates = nc::cumsum(intervals) - 1;

	statePositions = nc::NdArray<double>(1, numStates);
	nc::NdArray<int> stateIntervals = nc::NdArray<int>(1, numStates);

	nc::NdArray<double> slope;
	int idx = 0;
	for (auto i = 0 ; i < (int) intervals.size() ; ++i)
	{
		slope = nc::linspace(0.0, 1.0, intervals[i], false);
		std::copy(slope.begin(), slope.end(), statePositions.begin()+idx);
		idx += intervals[i];
	}
}

StateSpace::~StateSpace(){}


TransitionModel::TransitionModel(StateSpace stateSpace)
{
	nc::NdArray<int> states = nc::arange(stateSpace.numStates);
	states = nc::setdiff1d(states, stateSpace.firstStates);
	nc::NdArray<int> prevStates = states - 1;

	nc::NdArray<int> fromStates = stateSpace.lastStates;
	nc::NdArray<int> toStates = stateSpace.firstStates;
	nc::NdArray<int> fromInt = stateSpace.intervals;
	nc::NdArray<int> toInt = stateSpace.intervals;

	nc::NdArray<double> probabilities = nc::NdArray<double>(states.shape()).ones();
	nc::NdArray<double> prob = exponentialTransition(fromInt, toInt, transitionLambda);
	probabilities = nc::hstack({probabilities, prob.flatten()[prob.flatnonzero()]});

	std::pair<nc::NdArray<nc::uint32>, nc::NdArray<nc::uint32> > pair = prob.nonzero(); 	
	nc::NdArray<nc::uint32> fromProb = pair.first;
	nc::NdArray<nc::uint32> toProb = pair.second; 

	states = nc::hstack({states, toStates[toProb]});
	prevStates = nc::hstack({prevStates, fromStates[fromProb]});
	numStates = nc::max(prevStates)[0] + 1;

	nc::NdArray<double> transitionMatrix = nc::NdArray<double>(numStates).zeros();

	for (auto i = 0 ; i < (int) states.size() ; ++i)
	{
		transitionMatrix(states[i], prevStates[i]) = probabilities[i];
	}

	pair = transitionMatrix.nonzero(); 	
	nc::NdArray<nc::uint32> rowIndeces = pair.first;	
	tmIndices = pair.second;
	tmProbabilities = nc::log(probabilities);

	tmPointers = nc::NdArray<nc::uint32>(1, transitionMatrix.shape().rows + 1);
	int count = 0;
	tmPointers[0] = 0;
	for (auto r = 0 ; r < (int) transitionMatrix.shape().rows ; ++r)
	{
		count += transitionMatrix.row(r).flatnonzero().size();
		tmPointers[r+1] = count;
	}

	initialDistribution = nc::NdArray<double>(1, stateSpace.numStates).ones() / (double) stateSpace.numStates;
}


TransitionModel::~TransitionModel(){}


nc::NdArray<double> TransitionModel::exponentialTransition(nc::NdArray<int> &fromInt, 
		nc::NdArray<int> &toInt, double &transitionLambda)
{
	nc::NdArray<double> ratio(fromInt.size(), toInt.size());

	for (int i = 0 ; i < (int) fromInt.size() ; ++i)
	{
		for (int j = 0 ; j < (int) toInt.size() ; ++j)
		{
			ratio(i, j) = toInt[j] / (double) fromInt[i];
		}
	}

	nc::NdArray<double> prob = nc::exp(-transitionLambda * nc::abs(ratio - 1.0)).astype<double>();
	nc::NdArray<double> colSums = nc::sum(prob, nc::Axis::COL);

	for (int i = 0 ; i < (int) prob.shape().rows ; ++i)
	{
		for (int j = 0 ; j < (int) prob.shape().cols ; ++j)
		{
			if (prob(i, j) < NP_SPACING_1)
				prob(i, j) = 0.0;
			// normalize the emission probabilities
			prob(i, j) /= colSums[i];
		}
	}      		
   	return prob;
}


ObservationModel::ObservationModel(StateSpace &stateSpace, nc::NdArray<double> &observations)
{
	omPointers = nc::NdArray<nc::uint32>(1, stateSpace.numStates).zeros();

    // unless they are in the beat range of the state space
    double border = 1.0 / observationLambda;

    nc::NdArray<nc::uint32> ind = (border - stateSpace.statePositions).clip(0.0, 
    	std::numeric_limits<double>::max()).flatnonzero();
    for (auto i = 0 ; i < (int) ind.size() ; ++i)
    {
    	omPointers[ind[i]] = 1;
    }
      
    omDensities = nc::astype<double>(nc::log((1.0 - observations) / ((double) observationLambda - 1.0)));
    omDensities = nc::vstack({omDensities, nc::astype<double>(nc::log(observations))});
}


ObservationModel::~ObservationModel(){}


nc::NdArray<nc::uint32> viterbi(nc::NdArray<double> &observations, TransitionModel &tm, ObservationModel &om)
{
	unsigned long int numObservations = observations.size();
	unsigned long int state;
    unsigned long int prevState;
    double density;
    double transitionProb;

	nc::NdArray<double> currentViterbi = nc::NdArray<double>(1, tm.numStates); 
	nc::NdArray<double> previousViterbi = nc::log(tm.initialDistribution).astype<double>(); 
    nc::NdArray<nc::uint32> btPointers = nc::NdArray<nc::uint32>(numObservations, tm.numStates).zeros();

    for (auto i = 0 ; i < (int) numObservations ; ++i)
    {
    	for (auto s = 0 ; s < (int) tm.numStates ; ++s)
    	{
    		currentViterbi[s] = -std::numeric_limits<double>::infinity();;
    		density = om.omDensities(om.omPointers[s], i);

    		for (auto p = tm.tmPointers[s] ; p < tm.tmPointers[s + 1] ; ++p)
    		{
    			prevState = tm.tmIndices[p];

    			transitionProb = previousViterbi[prevState] + tm.tmProbabilities[p] + density;

    			if (transitionProb > currentViterbi[s])
    			{
    				currentViterbi[s] = transitionProb;
    				btPointers(i, s) = prevState;
    			}
    		}
    	}
    	previousViterbi = currentViterbi;
    }
    state = currentViterbi.argmax(nc::Axis::ROW)[0]; // ?

    // double logProbability = currentViterbi[state];
    // if np.isinf(log_probability):
    // warnings.warn('-inf log probability during Viterbi decoding '
    //               'cannot find a valid path', RuntimeWarning)
    // return np.empty(0, dtype=np.uint32), log_probability

   	nc::NdArray<nc::uint32> path(1, numObservations);

   	for (auto i = numObservations-1; i > 0 ; i -= 1)
    {
    	path[i] = state;
    	state = btPointers(i, state);
    }
    path[0] = state;
	return path;
}


std::vector<double> activationsToBeats(std::vector<double> &activations)
{
	nc::NdArray<double> observations(activations);
    nc::NdArray<double> beats;

    nc::NdArray<nc::uint32> first = {0};
    nc::NdArray<nc::uint32> last;

    // use only the activations > threshold
    if (threshold > 0.0)
    {
		nc::NdArray<nc::uint32> ind = (observations - (double) threshold).clip(0.0, 
    		std::numeric_limits<double>::max()).flatnonzero();

        if (ind.any()[0])
        {
            first = nc::maximum(first, nc::min(ind));
            last = nc::minimum({observations.size()}, nc::max(ind) + 1u);
        }
        else
        {
            last = first;
        }
        observations = observations[nc::Slice(first[0], last[0])];    
    }

	StateSpace stateSpace;
	TransitionModel transitionModel(stateSpace);
	ObservationModel observationModel(stateSpace, observations);

	nc::NdArray<nc::uint32> path = viterbi(observations, transitionModel, observationModel);

    nc::NdArray<nc::uint32> beatRange;
	nc::NdArray<nc::uint32> idx;
	nc::NdArray<nc::uint32> leftRight;
	nc::NdArray<nc::int32> left;
	nc::NdArray<nc::int32> right;
	double peak;

    // for each detection determine the "beat range", i.e. states where
    // the pointers of the observation model are 1
    beatRange = observationModel.omPointers[path];

    // get all change points between True and False
    idx = nc::flatnonzero(nc::diff(beatRange));
    idx = idx + nc::ones_like(idx);
    // if the first frame is in the beat range, add a change at frame 0
    if (beatRange[0] == 1)
        idx = nc::hstack({{0}, idx});
    // // if the last frame is in the beat range, append the length of the array
    if (beatRange[*beatRange.end() == 1])
       	idx = nc::hstack({idx, {beatRange.size()}});
    // iterate over all regions
    if (idx.any()[0])
    {
    	if (idx.size() % 2 != 0)
    		idx = nc::hstack({idx, {*(idx.end()-1)}});
    	leftRight = idx.reshape(-1, 2);
    	left = leftRight(leftRight.rSlice(), 0).astype<nc::int32>();
		right = leftRight(leftRight.rSlice(), 1).astype<nc::int32>();
		left = nc::maximum(nc::zeros_like(left), left - (right - left));

        for (int i = 0 ; i < (int) left.size() ; ++i)
        {
            peak = nc::argmax(observations[nc::Slice(left[i], right[i])])[0] + left[i];                	
            beats = nc::hstack({beats, {peak}});
        }
    }

    beats = (beats + (double) first[0]) / (double) fps;

    std::vector<double> returnVec = beats.astype<double>().toStlVector();
    return returnVec;    
}
