# coding: utf-8

import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np
cimport cython

from numpy.math cimport INFINITY

ctypedef np.uint32_t uint32_t


# State space

fps = 100

min_bpm = 56
max_bpm = 215

min_interval = int(60. * fps / max_bpm)
max_interval = int(60. * fps / min_bpm)

intervals = np.arange(min_interval, max_interval + 1)

num_states = int(np.sum(intervals))
num_intervals = len(intervals)

first_states = np.cumsum(np.r_[0, intervals[:-1]])
first_states = first_states.astype(np.int)

last_states = np.cumsum(intervals) - 1

state_positions = np.empty(num_states)
state_intervals = np.empty(num_states, dtype=np.int)

idx = 0
for i in intervals:
    state_positions[idx: idx + i] = np.linspace(0, 1, i, endpoint=False)
    state_intervals[idx: idx + i] = i
    idx += i


# Transition model

def exponential_transition(from_intervals, to_intervals, transition_lambda):
    
    ratio = to_intervals / from_intervals[:, np.newaxis]
    prob = np.exp(-transition_lambda * abs(ratio - 1.))
    
    # set values below threshold to 0
    prob[prob <= np.spacing(1)] = 0
    
    # normalize the emission probabilities
    prob /= np.sum(prob, axis=1)[:, np.newaxis]
    
    return prob


def make_sparse(states, prev_states, probabilities):
    # check for a proper probability distribution, i.e. the emission
    # probabilities of each prev_state must sum to 1
    
    if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
        raise ValueError('Not a probability distribution.')
        
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

states = np.arange(num_states, dtype=np.uint32)

# The probabiity of advancing to the next state is 1 
# except the transition from the last_state in a beat 
states = np.setdiff1d(states, first_states)
prev_states = states - 1
probabilities = np.ones_like(states, dtype=np.float)

# tempo transitions occur at the boundary between beats
from_states = last_states
to_states = first_states

from_int = state_intervals[from_states]
to_int = state_intervals[to_states]

prob = exponential_transition(from_int, to_int, transition_lambda)

# use only the states with transitions to/from != 0
from_prob, to_prob = np.nonzero(prob)

states = np.hstack((states, to_states[to_prob]))

prev_states = np.hstack((prev_states, from_states[from_prob]))
probabilities = np.hstack((probabilities, prob[prob != 0]))

tm_indices, tm_pointers, tm_probabilities = make_sparse(states, prev_states, probabilities)

tm_probabilities = np.log(tm_probabilities)

transition_model = (tm_indices, tm_pointers, tm_probabilities, num_states)


initial_distribution = np.ones(num_states, dtype=np.float) / num_states


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(observations, transition_model, observation_model, initial_distribution):
    
    _tm_indices, _tm_pointers, _tm_probabilities, _num_states = transition_model
    _om_pointers, _om_densities = observation_model
    
    cdef uint32_t [::1] tm_states = _tm_indices
    cdef uint32_t [::1] tm_pointers = _tm_pointers
    cdef double [::1] tm_probabilities = _tm_probabilities
    cdef unsigned int num_states = _num_states

    cdef unsigned int num_observations = len(observations)
    cdef uint32_t [::1] om_pointers = _om_pointers
    cdef double [:, ::1] om_densities = _om_densities

    cdef double [::1] current_viterbi = np.empty(num_states, dtype=np.float)

    cdef double [::1] previous_viterbi = np.log(initial_distribution)

    cdef uint32_t [:, ::1] bt_pointers = np.empty((num_observations,
                                                   num_states),
                                                  dtype=np.uint32)

    cdef unsigned int state, frame, prev_state, pointer
    cdef double density, transition_prob

    for frame in range(num_observations):

        for state in range(num_states):

            current_viterbi[state] = -INFINITY
            density = om_densities[frame, om_pointers[state]]

            for pointer in range(tm_pointers[state], tm_pointers[state + 1]):

                prev_state = tm_states[pointer]

                transition_prob = previous_viterbi[prev_state] + \
                                  tm_probabilities[pointer] + density

                if transition_prob > current_viterbi[state]:
                    current_viterbi[state] = transition_prob
                    bt_pointers[frame, state] = prev_state

        previous_viterbi[:] = current_viterbi

    state = np.asarray(current_viterbi).argmax()
    log_probability = current_viterbi[state]

    if np.isinf(log_probability):
        warnings.warn('-inf log probability during Viterbi decoding '
                      'cannot find a valid path', RuntimeWarning)
        return np.empty(0, dtype=np.uint32), log_probability

    path = np.empty(num_observations, dtype=np.uint32)

    for frame in range(num_observations -1, -1, -1):
        path[frame] = state
        state = bt_pointers[frame, state]

    # return the tracked path and its probability
    return path


threshold = 0.5
correct = True


def activations2beats(activations):
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


    # Observation model

    observation_lambda = 16

    observations = activations

    om_pointers = np.zeros(num_states, dtype=np.uint32)

    # unless they are in the beat range of the state space
    border = 1. / observation_lambda
        
    om_pointers[state_positions < border] = 1

    om_densities = np.empty((len(observations), 2), dtype=np.float)

    om_densities[:, 0] = np.log((1. - observations) /
                                 (observation_lambda - 1))
    om_densities[:, 1] = np.log(observations)

    observation_model = (om_pointers, om_densities)


    # Viterbi

    path = viterbi(observations, transition_model, observation_model, initial_distribution)

    if correct:
        # for each detection determine the "beat range", i.e. states where
        # the pointers of the observation model are 1
        beat_range = om_pointers[path]
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
                left = np.maximum(0,left - (right - left))
                
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
        beats = beats[om_pointers[path[beats]] == 1]
    # convert the detected beats to seconds and return them

    beats = (beats + first) / float(fps)

    return beats
    