__author__ = 'manshu'


import sys
import random
import math
import NaiveBayes as nb
import csv



def findMiError(predicted_class, actual_class, tuple_weights):
    error_mi = 0.0
    for i in range(0, len(predicted_class)):
        if predicted_class[i] != actual_class[i]:
            error_mi += tuple_weights[i] * 1
        else:
            error_mi += tuple_weights[i] * 0
    return error_mi

def assignNewTupleWeights(old_tuple_weights, eMi, predicted_class, actual_class):
    new_tuple_weights = []
    for i in range(0, len(predicted_class)):
        new_tup_i_weight = old_tuple_weights[i] * 1.0
        if predicted_class[i] == actual_class[i]:
            new_tup_i_weight = old_tuple_weights[i] * (eMi / (1.0 - eMi))
        new_tuple_weights.append(new_tup_i_weight)

    # Normalize new_tuples_weights
    old_tup_sum = sum(old_tuple_weights)
    new_tup_sum = sum(new_tuple_weights)

    if new_tup_sum == 0.0:
        return None

    normalized_tuple_weights = [tup_weight * (old_tup_sum / new_tup_sum) for tup_weight in new_tuple_weights]
    return normalized_tuple_weights

def makeNewTupleIds(tuple_weights, num_training_data):
    percentage_tuple_weights = [int(tup_weight * num_training_data) for tup_weight in tuple_weights]
    new_tuple_ids = []

    for i in range(0, num_training_data):
        new_tuple_ids += [i] * percentage_tuple_weights[i]

    return new_tuple_ids

def prefixScan(tuple_weights):
    new_tuple_weights = []
    prev_sum = 0.0
    for weight in tuple_weights:
        prev_sum += weight
        new_tuple_weights.append(prev_sum)
    return new_tuple_weights


def rangeSearch(weight_list, rand):
    lo = 0
    hi = len(weight_list)

    while hi >= lo:
        mid = (lo + hi) / 2

        if rand < weight_list[mid]:
            hi = mid - 1
        elif rand > weight_list[mid]:
            lo = mid + 1
        else:
            return mid
    return lo

def drawRandomPD(prefixed_tuple_weights):
    random_prob = random.random()
    return rangeSearch(prefixed_tuple_weights, random_prob)

def formEnsembleClassifiers(training_class, training_data, max_attribute_values, k, max_run=5):

    num_training_data = len(training_class)

    ensemble_classifiers = []
    errors_Mi = []

    tuple_weights = [(1.0 / len(training_data)) for i in range(0, len(training_data))]
    #tuple_ids = [i for i in range(0, len(training_class))]

    for rk in range(0, k):
        run_emis = []
        run_classifiers = []
        run_predictions = []
        run_training_class = []

        current_run = 0
        while True:
            new_training_data = []
            new_training_class = []
            prefixed_weights = prefixScan(tuple_weights)
            for i in range(0, num_training_data):
                pick_id = drawRandomPD(prefixed_weights)#random.choice(tuple_ids)
                new_training_data.append(training_data[pick_id])
                new_training_class.append(training_class[pick_id])

            Mi = nb.makeClassifier(new_training_class, new_training_data, max_attribute_values)
            predicted_class = nb.predictClass(new_training_data, Mi)

            eMi = findMiError(predicted_class, new_training_class, tuple_weights)

            if eMi < 0.5:# and (errors_Mi != [] and eMi < min(errors_Mi)):
                ensemble_classifiers.append(Mi)
                errors_Mi.append(eMi)
                break

            run_emis.append(eMi)
            run_classifiers.append(Mi)
            run_predictions.append(predicted_class)
            run_training_class.append(new_training_class)

            current_run += 1
            #
            # if current_run == max_run:
            #     eMi = max(run_emis)
            #     min_run_id = run_emis.index(eMi)
            #     ensemble_classifiers.append(run_classifiers[min_run_id])
            #     predicted_class = run_predictions[min_run_id]
            #     new_training_class = run_training_class[min_run_id]
            #     errors_Mi.append(eMi)
            #     break

        print eMi

        new_tuple_weights = assignNewTupleWeights(tuple_weights, eMi, predicted_class, new_training_class)
        if new_tuple_weights == None:
            rk -= 1
            continue
        tuple_weights = new_tuple_weights
        #tuple_ids = makeNewTupleIds(tuple_weights, num_training_data)

    return ensemble_classifiers, errors_Mi

def ensembleClassify(test_data, test_class, kClassifiers, kClassifiers_errors):
    boosted_predicted_class = []
    k_predictions = []

    for i in range(0, k):
        k_predictions.append(nb.predictClass(test_data, kClassifiers[i]))

    for i in range(0, len(test_data)):
        vote_probability = {}
        for ki in range(0, k):
            if k_predictions[ki][i] not in vote_probability:
                vote_probability[k_predictions[ki][i]] = 0
            if kClassifiers_errors[ki] == 0.0:
                vote_probability[k_predictions[ki][i]] += 0
            else:
                vote_probability[k_predictions[ki][i]] += math.log((1.0 - kClassifiers_errors[ki]) / kClassifiers_errors[ki])

        max_vote = 0
        max_class = None
        for class_label in vote_probability:
            if vote_probability[class_label] > max_vote:
                max_vote = vote_probability[class_label]
                max_class = class_label

        boosted_predicted_class.append(max_class)

    if test_class != []:
        nb.generateMeasures(test_class, boosted_predicted_class)

    return boosted_predicted_class

def solveAssignment(training_file, test_file, k):

    # read data from both files as it is
    features, training_class, training_data = nb.readFile(training_file)
    f2, ref_ids, test_data = nb.readFile(test_file)

    # find max number of attributes in both the files
    max_attribute, max_attribute_values = nb.findMaxNumAttributes(training_data, test_data)

    # format training and test data which can be used by classifier
    # training_class, training_data = nb.formatData(training_data, max_attribute)
    # test_class, test_data = nb.formatData(test_data, max_attribute)

    # make k classifiers from training_data and class labels using ensemble method adaboost
    kClassifiers, kClassifiers_errors = formEnsembleClassifiers(training_class, training_data, max_attribute_values, k)
    # print kClassifiers_errors
    # predict using all the classifier built using adaboost on test data
    boosted_predicted_class = ensembleClassify(training_data, training_class, kClassifiers, kClassifiers_errors)
    boosted_predicted_test_class = ensembleClassify(test_data, [], kClassifiers, kClassifiers_errors)

    output = []
    for i in range(0, len(boosted_predicted_test_class)):
        output.append([ref_ids[i], boosted_predicted_test_class[i]])

    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print "Please run the file with a training file and a test file"
    #     sys.exit(1)
    training_file = "x.csv"#sys.argv[1]
    test_file = "y.csv"#sys.argv[2]

    k = 5
    solveAssignment(training_file, test_file, k)