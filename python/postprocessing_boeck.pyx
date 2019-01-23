# coding: utf-8

import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np
cimport cython

from numpy.math cimport INFINITY

ctypedef np.uint32_t uint32_t


# os.chdir("../python/")


## State space model 

fps = 100

min_bpm = 56
max_bpm = 215

min_interval = 60. * fps / max_bpm
max_interval = 60. * fps / min_bpm

intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)

num_intervals = None

if num_intervals is not None and num_intervals < len(intervals):
        # we must approach the number of intervals iteratively
    num_log_intervals = num_intervals
    intervals = []
    while len(intervals) < num_intervals:
        intervals = np.logspace(np.log2(min_interval),
                                np.log2(max_interval),
                                num_log_intervals, base=2)
        # quantize to integer intervals
        intervals = np.unique(np.round(intervals))
        num_log_intervals += 1
        
intervals = np.ascontiguousarray(intervals, dtype=np.int)

# number of states and intervals
num_states = int(np.sum(intervals))
num_intervals = len(intervals)

# define first and last states
first_states = np.cumsum(np.r_[0, intervals[:-1]])
first_states = first_states.astype(np.int)

last_states = np.cumsum(intervals) - 1

# define the positions and intervals of the states
state_positions = np.empty(num_states)
state_intervals = np.empty(num_states, dtype=np.int)
# Note: having an index counter is faster than ndenumerate
idx = 0
for i in intervals:
    state_positions[idx: idx + i] = np.linspace(0, 1, i, endpoint=False)
    state_intervals[idx: idx + i] = i
    idx += i


## Transition model

def exponential_transition(from_intervals, to_intervals, transition_lambda,
                           threshold=np.spacing(1), norm=True):
    # no transition lambda
    if transition_lambda is None:
        # return a diagonal matrix
        return np.diag(np.diag(np.ones((len(from_intervals),
                                        len(to_intervals)))))
    # compute the transition probabilities
    ratio = (to_intervals.astype(np.float) /
             from_intervals.astype(np.float)[:, np.newaxis])
    prob = np.exp(-transition_lambda * abs(ratio - 1.))
    # set values below threshold to 0
    prob[prob <= threshold] = 0
    # normalize the emission probabilities
    if norm:
        prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob


def make_sparse(states, prev_states, probabilities):
    # check for a proper probability distribution, i.e. the emission
    # probabilities of each prev_state must sum to 1
    states = np.asarray(states)
    prev_states = np.asarray(prev_states, dtype=np.int)
    probabilities = np.asarray(probabilities)
    if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
        raise ValueError('Not a probability distribution.')
    # convert everything into a sparse CSR matrix, make sure it is square.
    # looking through prev_states is enough, because there *must* be a
    # transition *from* every state
    num_states = max(prev_states) + 1
    transitions = csr_matrix((probabilities, (states, prev_states)),
                             shape=(num_states, num_states))
    # convert to correct types
    states = transitions.indices.astype(np.uint32)
    pointers = transitions.indptr.astype(np.uint32)
    probabilities = transitions.data.astype(dtype=np.float)
    # return them
    return states, pointers, probabilities


transition_lambda = 100.0

# same tempo transitions probabilities within the state space is 1
# Note: use all states, but remove all first states because there are
#       no same tempo transitions into them
states = np.arange(num_states, dtype=np.uint32)
states = np.setdiff1d(states, first_states)
prev_states = states - 1
probabilities = np.ones_like(states, dtype=np.float)

# tempo transitions occur at the boundary between beats
# Note: connect the beat state space with itself, the transitions from
#       the last states to the first states follow an exponential tempo
#       transition (with the tempi given as intervals)
to_states = first_states
from_states = last_states
from_int = state_intervals[from_states]
to_int = state_intervals[to_states]

prob = exponential_transition(from_int, to_int, transition_lambda)

# use only the states with transitions to/from != 0
from_prob, to_prob = np.nonzero(prob)

states = np.hstack((states, to_states[to_prob]))

prev_states = np.hstack((prev_states, from_states[from_prob]))
probabilities = np.hstack((probabilities, prob[prob != 0]))

# make the transitions sparse
transitions = make_sparse(states, prev_states, probabilities)

states, pointers_tm, probabilities = zip(transitions)

log_probabilities = np.log(probabilities)


# ## Observation model

# In[73]:





# ## Hidden Markov model


initial_distribution = np.ones(num_states, dtype=np.float) / num_states


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(observations, states, pointers_tm, pointers_om, log_probabilities, 
            num_states, log_densities, initial_distribution):
    # transition model stuff
    cdef uint32_t [::1] tm_states = states
    cdef uint32_t [::1] tm_pointers = pointers_tm
    cdef double [::1] tm_probabilities = log_probabilities
    cdef unsigned int tm_num_states = num_states

    # observation model stuff
    cdef unsigned int num_observations = len(observations)
    cdef uint32_t [::1] om_pointers = pointers_om
    cdef double [:, ::1] om_densities = log_densities

    # current viterbi variables
    cdef double [::1] current_viterbi = np.empty(num_states, dtype=np.float)

    # previous viterbi variables, init with the initial state distribution
    cdef double [::1] previous_viterbi = np.log(initial_distribution)

    # back-tracking pointers
    cdef uint32_t [:, ::1] bt_pointers = np.empty((num_observations,
                                                   num_states),
                                                  dtype=np.uint32)
    # define counters etc.
    cdef unsigned int state, frame, prev_state, pointer
    cdef double density, transition_prob

    # iterate over all observations
    for frame in range(num_observations):
        # search for the best transition
        for state in range(num_states):
            # reset the current viterbi variable
            current_viterbi[state] = -INFINITY
            # get the observation model probability density value
            # the om_pointers array holds pointers to the correct
            # observation probability density value for the actual state
            # (i.e. column in the om_densities array)
            # Note: defining density here gives a 5% speed-up!?
            density = om_densities[frame, om_pointers[state]]
            # iterate over all possible previous states
            # the tm_pointers array holds pointers to the states which are
            # stored in the tm_states array
            for pointer in range(tm_pointers[state],
                                 tm_pointers[state + 1]):
                # get the previous state
                prev_state = tm_states[pointer]
                # weight the previous state with the transition probability
                # and the current observation probability density
                transition_prob = previous_viterbi[prev_state] + \
                                  tm_probabilities[pointer] + density
                # if this transition probability is greater than the
                # current one, overwrite it and save the previous state
                # in the back tracking pointers
                if transition_prob > current_viterbi[state]:
                    # update the transition probability
                    current_viterbi[state] = transition_prob
                    # update the back tracking pointers
                    bt_pointers[frame, state] = prev_state

        # overwrite the old states with the current ones
        previous_viterbi[:] = current_viterbi

    # fetch the final best state
    state = np.asarray(current_viterbi).argmax()
    # set the path's probability to that of the best state
    log_probability = current_viterbi[state]

    # raise warning if the sequence has -inf probability
    if np.isinf(log_probability):
        warnings.warn('-inf log probability during Viterbi decoding '
                      'cannot find a valid path', RuntimeWarning)
        # return empty path sequence
        return np.empty(0, dtype=np.uint32), log_probability

    # back tracked path, a.k.a. path sequence
    path = np.empty(num_observations, dtype=np.uint32)
    # track the path backwards, start with the last frame and do not
    # include the pointer for frame 0, since it includes the transitions
    # to the prior distribution states
    for frame in range(num_observations -1, -1, -1):
        # save the state in the path
        path[frame] = state
        # fetch the next previous one
        state = bt_pointers[frame, state]

    # return the tracked path and its probability
    return path, log_probability


threshold = 0.5
correct = True


def activations2beats(activations):

    observation_lambda = 16

    pointers_om = np.zeros(num_states, dtype=np.uint32)

    # unless they are in the beat range of the state space
    border = 1. / observation_lambda
        
    pointers_om[state_positions < border] = 1


    log_densities = np.empty((len(activations), 2), dtype=np.float)
    # Note: it's faster to call np.log 2 times instead of once on the
    #       whole 2d array
    log_densities[:, 0] = np.log((1. - activations) /
                                 (observation_lambda - 1))
    log_densities[:, 1] = np.log(activations)


    beats = np.empty(0, dtype=np.int)
    first = 0
    # use only the activations > threshold
    if threshold:
        idx = np.nonzero(activations >= threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    # return the beats if no activations given / remain after thresholding
    # if not activations.any():
    #     return beats

    path, _ = viterbi(activations, states[0], pointers_tm[0], pointers_om, log_probabilities[0], 
                      num_states, log_densities, initial_distribution)

    if correct:
        # for each detection determine the "beat range", i.e. states where
        # the pointers of the observation model are 1
        beat_range = pointers_om[path]
        # get all change points between True and False
        idx = np.nonzero(np.diff(beat_range))[0] + 1
        # if the first frame is in the beat range, add a change at frame 0
        if beat_range[0]:
            idx = np.r_[0, idx]
        # if the last frame is in the beat range, append the length of the
        # array
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        # iterate over all regions
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                # pick the frame with the highest activations value
                peak = np.argmax(activations[left:right]) + left
                beats = np.hstack((beats, peak))
    else:
        # just take the frames with the smallest beat state values
        from scipy.signal import argrelmin
        beats = argrelmin(state_positions[path], mode='wrap')[0]
        # recheck if they are within the "beat range", i.e. the pointers
        # of the observation model for that state must be 1
        # Note: interpolation and alignment of the beats to be at state 0
        #       does not improve results over this simple method
        beats = beats[pointers_om[path[beats]] == 1]
    # convert the detected beats to seconds and return them

    beats = (beats + first) / float(fps)

    return beats
    