import numpy as np
import warnings

def onset_evaluation(detections, annotations, window=0.025):
    # make sure the arrays have the correct types and dimensions
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')

    # init TP, FP, FN and errors
    tp = np.zeros(0)
    fp = np.zeros(0)
    tn = np.zeros(0)  # we will not alter this array
    fn = np.zeros(0)
    errors = np.zeros(0)

    # if neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        # return the arrays as is
        return tp, fp, tn, fn, errors
    # if only detections are given
    elif len(annotations) == 0:
        # all detections are FP
        return tp, detections, tn, fn, errors
    # if only annotations are given
    elif len(detections) == 0:
        # all annotations are FN
        return tp, fp, tn, annotations, errors

    # window must be greater than 0
    if float(window) <= 0:
        raise ValueError('window must be greater than 0')

    # sort the detections and annotations
    det = np.sort(detections)
    ann = np.sort(annotations)
    # cache variables
    det_length = len(detections)
    ann_length = len(annotations)
    det_index = 0
    ann_index = 0
    # iterate over all detections and annotations
    while det_index < det_length and ann_index < ann_length:
        # fetch the first detection
        d = det[det_index]
        # fetch the first annotation
        a = ann[ann_index]
        # compare them
        if abs(d - a) <= window:
            # TP detection
            tp = np.append(tp, d)
            # append the error to the array
            errors = np.append(errors, d - a)
            # increase the detection and annotation index
            det_index += 1
            ann_index += 1
        elif d < a:
            # FP detection
            fp = np.append(fp, d)
            # increase the detection index
            det_index += 1
            # do not increase the annotation index
        elif d > a:
            # we missed a annotation: FN
            fn = np.append(fn, a)
            # do not increase the detection index
            # increase the annotation index
            ann_index += 1
        else:
            # can't match detected with annotated onset
            raise AssertionError('can not match % with %', d, a)
    # the remaining detections are FP
    fp = np.append(fp, det[det_index:])
    # the remaining annotations are FN
    fn = np.append(fn, ann[ann_index:])
    # check calculations
    if len(tp) + len(fp) != len(detections):
        raise AssertionError('bad TP / FP calculation')
    if len(tp) + len(fn) != len(annotations):
        raise AssertionError('bad FN calculation')
    if len(tp) != len(errors):
        raise AssertionError('bad errors calculation')
    # convert to numpy arrays and return them
    return np.array(tp), np.array(fp), tn, np.array(fn), np.array(errors)


def num_tp(detections, annotations):

    return len(onset_evaluation(detections, annotations)[0])


def num_fp(detections, annotations):

    return len(onset_evaluation(detections, annotations)[1])


def num_tn(detections, annotations):

    return len(onset_evaluation(detections, annotations)[2])


def num_fn(detections, annotations):

    return len(onset_evaluation(detections, annotations)[3])


def precision(detections, annotations):
    # correct / retrieved
    retrieved = float(num_tp(detections, annotations) + num_fp(detections, annotations))
    # if there are no positive predictions, none of them are wrong
    if retrieved == 0:
        return 1.
    return num_tp(detections, annotations) / retrieved


def recall(detections, annotations):
    """Recall."""
    # correct / relevant
    relevant = float(num_tp(detections, annotations) + num_fn(detections, annotations))
    # if there are no positive annotations, we recalled all of them
    if relevant == 0:
        return 1.
    return num_tp(detections, annotations) / relevant


def fmeasure(detections, annotations):
    """F-measure."""
    # 2pr / (p+r)
    numerator = 2. * precision(detections, annotations) * recall(detections, annotations)
    if numerator == 0:
        return 0.
    return numerator / (precision(detections, annotations) + recall(detections, annotations))


def accuracy(detections, annotations):
    """Accuracy."""
    # acc: (TP + TN) / (TP + FP + TN + FN)
    denominator = num_fp(detections, annotations) + num_fn(detections, annotations) \
        + num_tp(detections, annotations) + num_tn(detections, annotations)
    if denominator == 0:
        return 1.
    numerator = float(num_tp(detections, annotations) + num_tn(detections, annotations))
    if numerator == 0:
        return 0.
    return numerator / denominator


def calc_absolute_errors(detections, annotations, matches=None):
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # return the errors
    return np.abs(calc_errors(detections, annotations, matches))


def calc_errors(detections, annotations, matches=None):
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calc error relative to those annotations
    errors = detections - annotations[matches]
    # return the errors
    return errors


def find_closest_matches(detections, annotations):
    
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        return np.zeros(0, dtype=np.int)

    if len(annotations) == 1:
        # return an array as long as the detections with indices 0
        return np.zeros(len(detections), dtype=np.int)

    indices = annotations.searchsorted(detections)
    indices = np.clip(indices, 1, len(annotations) - 1)
    left = annotations[indices - 1]
    right = annotations[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def pscore(detections, annotations, tolerance=0.2):
    # at least 2 annotations must be given to calculate an interval
    if len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations are needed for"
                                "P-Score.")
    # tolerance must be greater than 0
    if float(tolerance) <= 0:
        raise ValueError("`tolerance` must be greater than 0.")
    # the error window is the given fraction of the median beat interval
    window = tolerance * np.median(np.diff(annotations))
    # errors
    errors = calc_absolute_errors(detections, annotations)
    # count the instances where the error is smaller or equal than the window
    p = len(detections[errors <= window])
    # normalize by the max number of detections/annotations
    p /= float(max(len(detections), len(annotations)))
    # return p-score
    return p


def continuity(detections, annotations,
               offbeat=True, double=True, triple=True):
    # neither detections nor annotations are given
    if len(detections) == 0 and len(annotations) == 0:
        return 1., 1., 1., 1.
    # either a single beat detections or annotations given, score 0
    if len(detections) <= 1 or len(annotations) <= 1:
        return 0., 0., 0., 0.
    # evaluate the correct tempo
    cmlc, cmlt = cml(detections, annotations)
    amlc = cmlc
    amlt = cmlt
    # speed up calculation by skipping other metrical levels if the score is
    # higher than 0.5 already. We must have tested the correct metrical level
    # already, otherwise the cmlc score would be lower.
    if cmlc > 0.5:
        return cmlc, cmlt, amlc, amlt
    # create different variants of the annotations:
    # Note: double also includes half as does triple third, respectively
    sequences = variations(annotations, offbeat=offbeat, double=double,
                           half=double, triple=triple, third=triple)
    # evaluate these metrical variants
    for sequence in sequences:
        # if other metrical levels achieve higher accuracies, take these values
        try:
            # Note: catch the IntervalError here, because the beat variants
            #       could be too short for valid interval calculation;
            #       ok, since we already have valid values for amlc & amlt
            c, t = cml(detections, sequence)
        except BeatIntervalError:
            c, t = np.nan, np.nan
        amlc = max(amlc, c)
        amlt = max(amlt, t)
    # return a tuple
    return cmlc, cmlt, amlc, amlt


def cml(detections, annotations):
    
    phase_tolerance = 0.175
    tempo_tolerance = 0.175

    # at least 2 annotations must be given to calculate an interval
    if len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations are needed for "
                                "continuity scores, %s given." % annotations)
    # TODO: remove this, see TODO below
    if len(detections) < 2:
        raise BeatIntervalError("At least 2 detections are needed for "
                                "continuity scores, %s given." % detections)
    # determine closest annotations to detections
    closest = find_closest_matches(detections, annotations)
    # errors of the detections wrt. to the annotations
    errors = calc_absolute_errors(detections, annotations, closest)
    # detection intervals
    det_interval = calc_intervals(detections)
    # annotation intervals (get those intervals at the correct positions)
    ann_interval = calc_intervals(annotations)[closest]
    # a detection is correct, if it fulfills 2 conditions:
    # 1) must match an annotation within a certain tolerance window, i.e. the
    #    phase must be correct
    correct_phase = detections[errors <= ann_interval * phase_tolerance]
    # Note: the initially cited technical report has an additional condition
    #       ii) on page 5 which requires the same condition to be true for the
    #       previous detection / annotation combination. We do not enforce
    #       this, since a) this condition is kind of pointless: why shouldn't
    #       we count a correct beat just because its predecessor is not? and
    #       b) the original Matlab implementation does not enforce it either
    # 2) the tempo, i.e. the intervals, must be within the tempo tolerance
    # TODO: as agreed with Matthew, this should only be enforced from the 2nd
    #       beat onwards.
    correct_tempo = detections[abs(1 - (det_interval / ann_interval)) <=
                               tempo_tolerance]
    # combine the conditions
    correct = np.intersect1d(correct_phase, correct_tempo)
    # convert to indices
    correct_idx = np.searchsorted(detections, correct)
    # cmlc: longest continuous segment of detections normalized by the max.
    #       length of both sequences (detection and annotations)
    length = float(max(len(detections), len(annotations)))
    longest, _ = find_longest_continuous_segment(correct_idx)
    cmlc = longest / length
    # cmlt: same but for all detections (no need for continuity)
    cmlt = len(correct) / length
    # return a tuple
    return cmlc, cmlt


def find_closest_matches(detections, annotations):
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # TODO: right now, it only works with 1D arrays
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.int)
    # if only a single annotation is given
    if len(annotations) == 1:
        # return an array as long as the detections with indices 0
        return np.zeros(len(detections), dtype=np.int)
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = annotations.searchsorted(detections)
    indices = np.clip(indices, 1, len(annotations) - 1)
    left = annotations[indices - 1]
    right = annotations[indices]
    indices -= detections - left < right - detections
    # return the indices of the closest matches
    return indices


def calc_intervals(events, fwd=False):
    # at least 2 events must be given to calculate an interval
    if len(events) < 2:
        raise BeatIntervalError
    interval = np.zeros_like(events)
    if fwd:
        interval[:-1] = np.diff(events)
        # set the last interval to the same value as the second last
        interval[-1] = interval[-2]
    else:
        interval[1:] = np.diff(events)
        # set the first interval to the same value as the second
        interval[0] = interval[1]
    # return
    return interval


def find_longest_continuous_segment(sequence_indices):
    # continuous segments have consecutive indices, i.e. diffs =! 1 are
    # boundaries between continuous segments; add 1 to get the correct index
    boundaries = np.nonzero(np.diff(sequence_indices) != 1)[0] + 1
    # add a start (index 0) and stop (length of correct detections) to the
    # segment boundary indices
    boundaries = np.concatenate(([0], boundaries, [len(sequence_indices)]))
    # lengths of the individual segments
    segment_lengths = np.diff(boundaries)
    # return the length and start position of the longest continuous segment
    length = int(np.max(segment_lengths))
    start_pos = int(boundaries[np.argmax(segment_lengths)])
    return length, start_pos


def variations(sequence, offbeat=False, double=False, half=False,
               triple=False, third=False):
    # create different variants of the annotations
    sequences = []
    # double/half and offbeat variation
    if double or offbeat:
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            double_sequence = []
        else:
            # create a sequence with double tempo
            same = np.arange(0, len(sequence))
            # request one item less, otherwise we would extrapolate
            shifted = np.arange(0, len(sequence), 0.5)[:-1]
            double_sequence = np.interp(shifted, same, sequence)
        # same tempo, half tempo off
        if offbeat:
            sequences.append(double_sequence[1::2])
        # double/half tempo variations
        if double:
            # double tempo
            sequences.append(double_sequence)
    if half:
        # half tempo odd beats (i.e. 1,3,1,3,..)
        sequences.append(sequence[0::2])
        # half tempo even beats (i.e. 2,4,2,4,..)
        sequences.append(sequence[1::2])
    # triple/third tempo variations
    if triple:
        if len(sequence) == 0:
            # if we have an empty sequence, there's nothing to interpolate
            triple_sequence = []
        else:
            # create a annotation sequence with triple tempo
            same = np.arange(0, len(sequence))
            # request two items less, otherwise we would extrapolate
            shifted = np.arange(0, len(sequence), 1. / 3)[:-2]
            triple_sequence = np.interp(shifted, same, sequence)
        # triple tempo
        sequences.append(triple_sequence)
    if third:
        # third tempo 1st beat (1,4,3,2,..)
        sequences.append(sequence[0::3])
        # third tempo 2nd beat (2,1,4,3,..)
        sequences.append(sequence[1::3])
        # third tempo 3rd beat (3,2,1,4,..)
        sequences.append(sequence[2::3])
    # return
    return sequences


def information_gain(detections, annotations):   
    num_bins = 40
    # neither detections nor annotations are given, perfect score
    if len(detections) == 0 and len(annotations) == 0:
        # return a max. information gain and an empty error histogram
        return np.log2(num_bins), np.zeros(num_bins)
    # either beat detections or annotations are empty, score 0
    # Note: use "or" here since we test both the detections against the
    #       annotations and vice versa during the evaluation process
    if len(detections) <= 1 or len(annotations) <= 1:
        # return an information gain of 0 and a uniform beat error histogram
        # Note: because swapped detections and annotations should return the
        #       same uniform histogram, the maximum length of the detections
        #       and annotations is chosen (instead of just the length of the
        #       annotations as in the Matlab implementation).
        max_length = max(len(detections), len(annotations))
        return 0., np.ones(num_bins) * max_length / float(num_bins)
    # at least 2 annotations must be given to calculate an interval
    if len(detections) < 2 or len(annotations) < 2:
        raise BeatIntervalError("At least 2 annotations and 2 detections are"
                                "needed for Information gain.")
    # check if there are enough beat annotations for the number of bins
    if num_bins > len(annotations):
        warnings.warn("Not enough beat annotations (%d) for %d histogram bins."
                      % (len(annotations), num_bins))
    # create bins edges for the error histogram
    histogram_bins = _histogram_bins(num_bins)
    # evaluate detections against annotations
    fwd_histogram = _error_histogram(detections, annotations, histogram_bins)
    fwd_ig = _information_gain(fwd_histogram)
    # if only a few (but correct) beats are detected, the errors could be small
    # thus evaluate also the annotations against the detections, i.e. simulate
    # a lot of false positive detections
    bwd_histogram = _error_histogram(annotations, detections, histogram_bins)
    bwd_ig = _information_gain(bwd_histogram)
    # only use the lower information gain
    if fwd_ig < bwd_ig:
        return fwd_ig, fwd_histogram
    return bwd_ig, bwd_histogram


def _histogram_bins(num_bins):
    # allow only even numbers and require at least 2 bins
    if num_bins % 2 != 0 or num_bins < 2:
        # Note: because of the implementation details of the histogram, the
        #       easiest way to make sure the an error of 0 is always mapped
        #       to the centre bin is to enforce an even number of bins
        raise ValueError("Number of error histogram bins must be even and "
                         "greater than 0")
    # since np.histogram accepts a sequence of bin edges we just increase the
    # number of bins by 1, but we need to apply offset
    offset = 0.5 / num_bins
    # because the histogram is made circular by adding the last bin to the
    # first one before being removed, increase the number of bins by 2
    return np.linspace(-0.5 - offset, 0.5 + offset, num_bins + 2)


def _error_histogram(detections, annotations, histogram_bins):
    # get the relative errors of the detections to the annotations
    errors = calc_relative_errors(detections, annotations)
    # map the relative beat errors to the range of -0.5..0.5
    errors = np.mod(errors + 0.5, -1) + 0.5
    # get bin counts for the given errors over the distribution
    histogram = np.histogram(errors, histogram_bins)[0].astype(np.float)
    # make the histogram circular by adding the last bin to the first one
    histogram[0] += histogram[-1]
    # return the histogram without the last bin
    return histogram[:-1]


def calc_relative_errors(detections, annotations, matches=None):
    # make sure the arrays have the correct types
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if matches is not None:
        matches = np.asarray(matches, dtype=np.int)
    # if no detections or annotations are given
    if len(detections) == 0 or len(annotations) == 0:
        # return a empty array
        return np.zeros(0, dtype=np.float)
    # determine the closest annotations
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    # calculate the absolute errors
    errors = calc_errors(detections, annotations, matches)
    # return the relative errors
    return np.abs(1 - (errors / annotations[matches]))


def _information_gain(error_histogram):
    # calculate the entropy of th error histogram
    if np.asarray(error_histogram).any():
        entropy = _entropy(error_histogram)
    else:
        # an empty error histogram has an entropy of 0
        entropy = 0.
    # return information gain
    return np.log2(len(error_histogram)) - entropy


def _entropy(error_histogram):
    # copy the error_histogram, because it must not be altered
    histogram = np.copy(error_histogram).astype(np.float)
    # normalize the histogram
    histogram /= np.sum(histogram)
    # set all 0 values to 1 to make entropy calculation well-behaved
    histogram[histogram == 0] = 1.
    # calculate entropy
    return - np.sum(histogram * np.log2(histogram))


