__author__ = 'manshu'

import sys
import math
import csv

numerical_cols = ['VehOdo',  'VehicleAge',   'VehBCost', 'WarrantyCost',
                  'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
                  'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
                  'ProfitAcquisitionAverage', 'ProfitAcquisitionClean', 'ProfitCurrentAverage', 'ProfitCurrentClean']

numerical_cols_id = []

def readFile(file):
    class_labels = []
    classifier_data = []
    with open(file, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        line = 0
        for row in csv_reader:
            line += 1
            if line == 1:
                features = row[1:]
                continue
            class_labels.append(int(row[0]))
            classifier_data.append({i : 0 for i in range(1, len(features) + 1)})
            data_tuple = row[1:]
            for i in range(0, len(data_tuple)):
                attr_val = data_tuple[i]
                classifier_data[-1][i] = attr_val #int(math.ceil(float(attr_val)))

    if numerical_cols_id == []:
        for i in range(0, len(features)):
            feature = features[i]
            if feature in numerical_cols:
                numerical_cols_id.append(i)

    for row in classifier_data:
        for attr in row:
            if attr in numerical_cols_id:
                row[attr] = int(math.ceil(float(row[attr])))
            else:
                row[attr] = int(row[attr])

    print features
    print numerical_cols_id
    #print class_labels
    print classifier_data[1:5]
    return features, class_labels, classifier_data

def findMaxNumAttributes(training_data, test_data):
    max_attribute = 0
    max_attribute_values = {}

    for row in (training_data + test_data):
        row_max_attribute = 0 # max([int(col.split(':')[0]) for col in data_instance[1:]])
        for col in row:
            # attribute_val_pair = col.split(':')
            # attribute = int(attribute_val_pair[0])
            # attribute_value = int(attribute_val_pair[1])
            attribute = col
            attribute_value = row[col]
            if attribute > row_max_attribute:
                row_max_attribute = attribute

            if attribute not in max_attribute_values:
                max_attribute_values[attribute] = 0

            if attribute_value > max_attribute_values[attribute]:
                max_attribute_values[attribute] = attribute_value

        if row_max_attribute > max_attribute:
            max_attribute = row_max_attribute

    for value in max_attribute_values:
        max_attribute_values[value] += 1

    return max_attribute, max_attribute_values

def formatData(input_data, max_attribute):
    classifier_data = []
    classifier_class = []

    for row in input_data:
        data_instance = row.split(' ')
        classifier_class.append(int(data_instance[0]))
        classifier_data.append({i : 0 for i in range(1, max_attribute + 1)})
        for col in data_instance[1:]:
            key_val_pair = col.split(':')
            classifier_data[-1][int(key_val_pair[0])] = int(key_val_pair[1])

    return classifier_class, classifier_data

def guassianProbability(guassian_attr_val_class, attr, xk, Ci):
    gxus = 1 / math.sqrt(2 * math.pi)

    if attr not in guassian_attr_val_class:
        return 0

    if Ci not in guassian_attr_val_class[attr]:
        return 0

    mean = guassian_attr_val_class[attr][Ci][0]
    std = guassian_attr_val_class[attr][Ci][1]

    if std == 0.0:
        std = 1.0
    gxus *= 1.0 / std
    epow = math.e ** (-1 * ((xk - mean) * (xk - mean)) / (2 * std * std))

    gxus *= epow

    return gxus

def makeGuassianCalculations(train_data, train_class):
    guassian_attr_val_class = {}

    for i in range(0, len(train_data)):
        data_instance = train_data[i]
        data_class = train_class[i]
        for attr in data_instance:
            if attr in numerical_cols_id:
                if attr not in guassian_attr_val_class:
                    guassian_attr_val_class[attr] = {}
                if data_class not in guassian_attr_val_class[attr]:
                    guassian_attr_val_class[attr][data_class] = [0.0, 0.0, 0]

                guassian_attr_val_class[attr][data_class][0] += data_instance[attr]
                guassian_attr_val_class[attr][data_class][2] += 1

    for attr in guassian_attr_val_class:
        for data_class in guassian_attr_val_class[attr]:
            sum = guassian_attr_val_class[attr][data_class][0]
            count = guassian_attr_val_class[attr][data_class][2]
            mean = sum / (1.0 * count)
            guassian_attr_val_class[attr][data_class][0] = mean

    for i in range(0, len(train_data)):
        data_instance = train_data[i]
        data_class = train_class[i]
        for attr in data_instance:
            if attr in guassian_attr_val_class:
                mean = guassian_attr_val_class[attr][data_class][0]
                guassian_attr_val_class[attr][data_class][1] += (data_instance[attr] - mean) * (data_instance[attr] - mean)

    for attr in guassian_attr_val_class:
        for data_class in guassian_attr_val_class[attr]:
            mean_sum = guassian_attr_val_class[attr][data_class][1]
            count = guassian_attr_val_class[attr][data_class][2]
            std = math.sqrt(mean_sum / (1.0 * count))
            guassian_attr_val_class[attr][data_class][1] = std

    return guassian_attr_val_class

def makeClassifier(training_class, training_data, max_attribute_values):
    probability_class = {}
    probability_attribute_val_class = {}
    num_samples = len(training_class)
    guassian_attr_val_class = makeGuassianCalculations(training_data, training_class)

    print "Guassian"
    print guassian_attr_val_class

    for class_label in training_class:
        if class_label not in probability_class:
            probability_class[class_label] = 0
        probability_class[class_label] += 1
    
    for attr in training_data[0].keys():
        if attr in numerical_cols_id:
            continue
        probability_attribute_val_class[attr] = {}
        for attr_value in range(0, max_attribute_values[attr]):
            probability_attribute_val_class[attr][attr_value] = {}
            for class_label in probability_class:
                probability_attribute_val_class[attr][attr_value][class_label] = 1

        
    for data_id in range(0, num_samples):
        data_instance = training_data[data_id]
        for attribute in data_instance:
            if attribute in numerical_cols_id:
                continue
            attribute_value = data_instance[attribute]
            # if attribute_value not in probability_attribute_val_class[attribute]:
            #     probability_attribute_val_class[attribute][attribute_value] = {}
            #     for class_label in probability_class: probability_attribute_val_class[attribute][attribute_value][class_label] = 0

            probability_attribute_val_class[attribute][attribute_value][training_class[data_id]] += 1

    for attribute in probability_attribute_val_class:
        if attribute in numerical_cols_id:
            continue
        for attribute_value in probability_attribute_val_class[attribute]:
            for class_label in probability_attribute_val_class[attribute][attribute_value]:
                probability_attribute_val_class[attribute][attribute_value][class_label] /= (1.0 * (probability_class[class_label] + max_attribute_values[attribute]))

    return [probability_class, probability_attribute_val_class, guassian_attr_val_class]

def predictClass(test_data, nBclassifier):
    probability_class = nBclassifier[0]
    probability_attribute_val_class = nBclassifier[1]
    guassian_attr_val_class = nBclassifier[2]

    predicted_class = []

    for data_instance in test_data:
        max_probability = 0.0
        max_class = probability_class.keys()[0]

        for class_label in probability_class:
            ci_probability = probability_class[class_label]

            for attribute in data_instance:
                attribute_value = data_instance[attribute]
                # if attribute_value not in probability_attribute_val_class[attribute]:
                #     ci_probability *= 0
                # else:
                if attribute in numerical_cols_id:
                    ci_probability *= guassianProbability(guassian_attr_val_class, attribute, attribute_value, class_label)
                else:
                    ci_probability *= probability_attribute_val_class[attribute][attribute_value][class_label]

            if ci_probability > max_probability:
                max_probability = ci_probability
                max_class = class_label

        predicted_class.append(max_class)

    return predicted_class

def generateMeasures(test_labels, predicted_labels):
    tp = tn = fp = fn = 0
    if not test_labels or not predicted_labels:
        print str(tp) + " " + str(fn) + " " + str(fp) + " " + str(tn)
        return

    for i in range(0, len(test_labels)):
        tvalue = test_labels[i]
        pvalue = predicted_labels[i]
        if tvalue == 1:
            if pvalue == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pvalue == 1:
                fp += 1
            else:
                tn += 1

    p = len([i for i in range(0, len(test_labels)) if test_labels[i] == 1])
    n = len([i for i in range(0, len(test_labels)) if test_labels[i] == 0])

    accuracy = (tp + tn) / (1.0 * (p + n))
    error_rate = (fp + fn) / (1.0 * (p + n))
    senstivity = tp / (1.0 * p)
    recall = senstivity
    specificity = tn / (1.0 * n)
    precision = tp / (1.0 * (tp + fp))
    f1_score = 2.0 * (precision * recall) / (1.0 * (precision + recall))
    B = 0.5
    fbscore1 = ((1 + B*B) * precision * recall) / (1.0 * (B*B*precision + recall))
    B = 2
    fbscore2 = ((1 + B*B) * precision * recall) / (1.0 * (B*B*precision + recall))


    print "Accuracy = " + str(accuracy) + ", Error Rate = " + str(error_rate) + ", Senstivity = " + str(senstivity) + ", Specificity = " + str(specificity)  + ", Precision = " + str(precision) + ", f1-score = " + str(f1_score) + ", Fb(0.5) = " + str(fbscore1) + ", Fb(2) = " + str(fbscore2)
    print str(tp) + " " + str(fn) + " " + str(fp) + " " + str(tn)

def solveAssignment(training_file, test_file):

    # read data from both files as it is
    features, training_class, training_data = readFile(training_file)

    f2, ref_ids, test_data = readFile(test_file)
    print "RefIds", ref_ids
    #
    # # find max number of attributes in both the files
    max_attribute, max_attribute_values = findMaxNumAttributes(training_data, test_data)
    print max_attribute
    print max_attribute_values
    #
    # # format training and test data which can be used by classifier
    # training_class, training_data = formatData(training_data, max_attribute)
    # test_class, test_data = formatData(test_data, max_attribute)
    #
    # # make classifier from training_data and class labels
    nBclassifier = makeClassifier(training_class, training_data, max_attribute_values)
    #
    # #print nBclassifier[0]
    # #print nBclassifier[1]
    #
    # predict using the classifier built on training and test data
    predicted_class = predictClass(training_data, nBclassifier)
    generateMeasures(training_class, predicted_class)

    predicted_class = predictClass(test_data, nBclassifier)
    print predicted_class

    output = []
    for i in range(0, len(predicted_class)):
        output.append([ref_ids[i], predicted_class[i]])

    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output)
    # generateMeasures(test_class, predicted_class)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print "Please run the file with a training file and a test file"
    #     sys.exit(1)
    training_file = "x.csv" #sys.argv[1]
    test_file = "y.csv" #sys.argv[2]
    solveAssignment(training_file, test_file)
