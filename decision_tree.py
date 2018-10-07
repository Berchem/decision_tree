#! python2.7
# -*-coding: utf-8-*-
from __future__ import division
from functools import partial
from collections import defaultdict, Counter
import math
import pprint
import csv


def entropy(probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2)
               for p in probabilities
               if p)  # ignore zero probabilities


def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets
    subsets is a list of lists of labeled data"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
                for subset in subsets)


def partition_by(inputs, attribute):
    """each input is a pair (attribute_dict, label).
    returns a dict : attribute_value -> inputs"""
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]  # get the value of the specified attribute
        groups[key].append(input)  # then add this input to the correct list
    return groups


def partition_entropy_by(inputs, attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0:
        return False  # no Trues? return a "False" leaf
    if num_falses == 0:
        return True  # no Falses? return a "True" leaf
    if not split_candidates:  # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf
    # otherwise, split on the best attribute
    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    # recursively build the subtrees
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.iteritems()}
    subtrees[None] = num_trues > num_falses  # default case
    return best_attribute, subtrees


def classify(tree, input):
    if tree in [True, False]:
        return tree
    attribute, subtree_dict = tree
    input_key = input.get(attribute)
    if input_key not in subtree_dict:
        subtree_key = None
    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)


def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    return Counter(votes).most_common(1)[0][0]


def read_csv(src, to_dict=True):
    with open(src) as f:
        data = list(csv.reader(f))

    if to_dict:
        attribute = data[0][:-1]
        records = data[1:]
        data_dict = list(
            map(
                lambda record: (dict(zip(attribute, record[:-1])), eval(record[-1])),
                records
            )
        )
        return data_dict

    else:
        return data
