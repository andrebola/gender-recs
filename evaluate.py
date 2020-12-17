from collections import OrderedDict
from collections import namedtuple

import numpy as np
from scipy import stats


def coverage(predicted, max_n_predictions=100):
    target_set = set()
    for i in range(len(predicted)):#.shape[0]):
        predictions = predicted[i][:max_n_predictions]
        target_set = target_set.union(predictions)
    return len(target_set)


def mapk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

# precision
def precision(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    target_set = set(targets)
    return float(len(set(predictions).intersection(target_set))) / max_n_predictions


"""
def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted.tolist())])
"""

# R precision
def r_precision(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    target_set = set(targets)
    target_count = len(target_set)
    return float(len(set(predictions[:target_count]).intersection(target_set))) / target_count

def dcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    """Compute the Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Returns:
        DCG value
    """
    #retrieved_elements = __get_unique(retrieved_elements[:k])
    #relevant_elements = __get_unique(relevant_elements)
    retrieved_elements = retrieved_elements[:k]
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))


def ndcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    r"""Compute the Normalized Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.
    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Returns:
        NDCG value
    """

    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    idcg = dcg(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg

methods = {'r-precision': r_precision,
        'map': mapk,
        'precision': precision,
        'ndcg': ndcg}

def evaluate(metrics, test, predicted):
    results = {}
    for i in range(predicted.shape[0]):
        if len(test[i]) > 0 :
            for metric_str in metrics:
                metric = metric_str.split('@')
                if len(metric) > 1:
                    res = methods[metric[0]](test[i], predicted[i], int(metric[1]))
                else:
                    res = methods[metric[0]](test[i], predicted[i])
                if metric_str not in results:
                    results[metric_str] = []
                results[metric_str].append(res)
    final_results= {}
    for res in results.keys():
        final_results[res] = stats.describe(results[res]).mean
    return final_results

