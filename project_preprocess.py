__author__ = 'manshu'

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from pandas import DataFrame
from pandas import Series
import math

remove_columns = ['RefId', 'VehYear',
                  'Nationality', 'Transmission', 'TopThreeAmericanName']

class_column = 'IsBadBuy'
missing_values = ['NaN', '', 'NULL', 'NOT AVAILABLE', 'NOT AVAIL']

other_cols = ['WheelTypeID', 'IsOnlineSale', 'BYRNO', 'VNZIP1']
nominal_cols = ['Auction', 'PurchDate', 'Make', 'Trim', 'Model', 'SubModel', 'Color', 'WheelType', 'VNST', 'Size', 'PRIMEUNIT', 'AUCGUART'] + other_cols

replicate_cols = {'WheelType': 3, 'VehBCost': 3, 'VehOdo': 1, 'BYRNO': 1, 'VehicleAge': 1, 'PRIMEUNIT': 1}#, 'MMRCurrentAuctionCleanPrice': 1, 'MMRCurrentRetailAveragePrice': 1}
#replica_factor = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1]

numerical_cols = ['VehOdo', 'VehicleAge', 'VehBCost', 'WarrantyCost', 'MMRCurrentAuctionCleanPrice',
                  'MMRCurrentRetailAveragePrice', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                  'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentRetailCleanPrice']

#create_new_attribute = []
create_new_attribute = [
    [['MMRAcquisitionRetailAveragePrice', 'MMRAcquisitionAuctionAveragePrice'], 'ProfitAcquisitionAverage', lambda x,y: x - y],
    [['MMRAcquisitonRetailCleanPrice', 'MMRAcquisitionAuctionCleanPrice'], 'ProfitAcquisitionClean', lambda x,y: x - y],
    [['MMRCurrentRetailAveragePrice', 'MMRCurrentAuctionAveragePrice'], 'ProfitCurrentAverage', lambda x,y: x - y],
    [['MMRCurrentRetailCleanPrice', 'MMRCurrentAuctionCleanPrice'], 'ProfitCurrentClean', lambda x,y: x - y],
    [['ProfitAcquisitionAverage', 'ProfitCurrentAverage'], 'AverageProfit', lambda x,y: x - y],
    [['ProfitAcquisitionClean', 'ProfitCurrentClean'], 'CleanProfit', lambda x,y: x - y],
    [['VehOdo', 'VehicleAge'], 'OdoPAge', lambda x,y: x/(1.0 * y)]

]
nominal_maps = {}

#neglect_cols = [0, 2, 5, 12, 28, 29, 30, 31, 6, 16, 13, 17, 15, 11, 32, 26, 27];
#bad_nominal_cols = [3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 26, 27, 32]
#bad_numerical_cols = [14, 18, 19, 20, 21, 22, 23, 24, 25, 33]

def replace_by_most_frequent(cdata, missing_value=0):
    data_count = {}
    for data in cdata:
        if data == missing_value:
            continue
        if data not in data_count:
            data_count[data] = 0
        data_count[data] += 1

    most_freq_value = max(data_count.values())
    most_freq_key = -1

    for key in data_count:
        if data_count[key] == most_freq_value:
            most_freq_key = key
            break

    # for i in range(0, len(cdata)):
    #     if cdata[i] == missing_value:
    #         cdata[i] = most_freq_key

    return most_freq_key

def replace_by_mean_and_normalize(cdata, missing_value=0):
    sum_data = 0
    mean_count = 0
    for i in range(0, len(cdata)):
        #cdata[i] = float(cdata[i])
        cdata[i] = int(cdata[i])
        data = int(cdata[i])
        if data != missing_value:
            sum_data += data
            mean_count += 1

    mean = sum_data / mean_count

    new_max = 0.0
    new_min = 0.0
    new_mean = 0.0
    for i in range(0, len(cdata)):
        if cdata[i] == missing_value:
            cdata[i] = int(mean)
        else:
            cdata[i] = int(cdata[i])
    #     if cdata[i] < new_min:
    #         new_min = cdata[i]
    #     if cdata[i] > new_max:
    #         new_max = cdata[i]
    #     new_mean += cdata[i]
    # #
    # for i in range(0, len(cdata)):
    #     cdata[i] = (cdata[i] - new_min) / (new_max - new_min)
    # std = 0.0
    # for i in range(0, len(cdata)):
    #     std += (cdata[i] - new_mean) * (cdata[i] - new_mean)
    #
    # std /= 1.0 * len(cdata)
    #
    # for i in range(0, len(cdata)):
    #     cdata[i] = (cdata[i] - new_mean) / std

    return mean

def preprocess(data_file):

    classifier_data = pd.read_csv(data_file, keep_default_na=False)

    for mv in missing_values:
        classifier_data.replace(mv, 'NaN', inplace=True)

    columns = list(classifier_data.columns.values)
    print columns


    class_data = classifier_data[class_column]
    for i in range(0, len(class_data)):
        class_data[i] = int(class_data[i])

    for column in columns:
        if column in remove_columns:# or column == class_column:
            classifier_data.drop(column, axis=1, inplace=True)


    num_attributes = len(classifier_data.columns)
    print "After removing redundant attributes = ", num_attributes

    #smooth_cols(classifier_data['Color'])


    imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0, copy=True)

    for col in nominal_cols:

        unique_col_data = list(np.unique(classifier_data[col]))

        if 'NaN' in unique_col_data:
            unique_col_data.remove('NaN')

        unique_col_length = len(unique_col_data)
        print unique_col_length, unique_col_data

        nominal_maps[col] = [{}, 0, 0]

        unique_val = 0
        for unique_attr_val in ['NaN'] + unique_col_data:
            nominal_maps[col][0][unique_attr_val] = unique_val
            unique_val += 1

        nominal_maps[col][1] = unique_val

        if col not in other_cols:
            classifier_data[col] = classifier_data[col].replace(to_replace=['NaN'] + unique_col_data, value=range(0, unique_col_length + 1))
        else:
            classifier_data[col].replace('NaN', 0, inplace=True)
            #del nominal_maps[col][0]['NaN']
            #nominal_maps[col][0][-1] = 0
            #classifier_data[col] = classifier_data[col].replace(to_replace=[-1] + unique_col_data, value=range(0, unique_col_length + 1))
        #most_frequent = replace_by_most_frequent(classifier_data[col])
        #nominal_maps[col][2] = most_frequent


    for col in numerical_cols:
        classifier_data[col].replace('NaN', 0, inplace=True)
        #print classifier_data[col]
        replace_by_mean_and_normalize(classifier_data[col])

    for i in range(0, len(create_new_attribute)):
        row = create_new_attribute[i]
        classifier_data[row[1]] = Series(np.random.randn(len(classifier_data[row[0][0]])), index=classifier_data.index)
        fx = row[2]
        for j in range(0, len(classifier_data[row[0][0]])):
            classifier_data[row[1]][j] = fx(classifier_data[row[0][0]][j], classifier_data[row[0][1]][j])

    for col_name in replicate_cols:
        #col_name = replicate_cols[i]
        for j in range(0, replicate_cols[col_name]):
            new_col = col_name + str(j)
            classifier_data[new_col] = Series(np.random.randn(len(classifier_data[col_name])), index=classifier_data.index)
            for k in range(0, len(classifier_data[col_name])):
                classifier_data[new_col][k] = classifier_data[col_name][k]

        #classifier_data.drop(row[0][0], axis=1, inplace=True)
        #classifier_data.drop(row[0][1], axis=1, inplace=True)

    #classifier_data.to_excel("x.xls")
    classifier_data.to_csv("x.csv", sep=',')

    classifier_data.drop(class_column, axis=1, inplace=True)

def convertTestNominalData(tdata, col, missing_value='NaN'):
    for i in range(0, len(tdata)):
        # if tdata[i] == missing_value:
        #     tdata[i] = nominal_maps[col][2]
        #     continue
        if tdata[i] in nominal_maps[col][0]:
            tdata[i] = nominal_maps[col][0][tdata[i]]
        else:
            nominal_maps[col][0][tdata[i]] = nominal_maps[col][1]
            tdata[i] = nominal_maps[col][1]
            nominal_maps[col][1] += 1

def convertTest(test_file, est=None):
    test_data = pd.read_csv(test_file, keep_default_na=False)

    for mv in missing_values:
        test_data.replace(mv, 'NaN', inplace=True)

    columns = list(test_data.columns.values)
    #print columns

    for column in columns:
        if column == "RefId":
            continue
        if column in remove_columns or column == class_column:
            test_data.drop(column, axis=1, inplace=True)


    num_attributes = len(test_data.columns)
    #print "After removing redundant attributes = ", num_attributes

    for col in nominal_cols:
        if col not in other_cols:
            convertTestNominalData(test_data[col], col)
        else:
            test_data[col].replace('NaN', 0, inplace=True)
            #convertTestNominalData(test_data[col], col, -1)
        #most_frequent = replace_by_most_frequent(test_data[col])

    for col in numerical_cols:
        test_data[col].replace('NaN', 0, inplace=True)
        #print classifier_data[col]
        replace_by_mean_and_normalize(test_data[col])

    for i in range(0, len(create_new_attribute)):
        row = create_new_attribute[i]
        test_data[row[1]] = Series(np.random.randn(len(test_data[row[0][0]])), index=test_data.index)
        fx = row[2]
        for j in range(0, len(test_data[row[0][0]])):
            test_data[row[1]][j] = fx(test_data[row[0][0]][j], test_data[row[0][1]][j])

        #test_data.drop(row[0][0], axis=1, inplace=True)
        #test_data.drop(row[0][1], axis=1, inplace=True)

    for col_name in replicate_cols:#range(0, len(replicate_cols)):
        #col_name = replicate_cols[i]
        for j in range(0, replicate_cols[col_name]):
            new_col = col_name + str(j)
            test_data[new_col] = Series(np.random.randn(len(test_data[col_name])), index=test_data.index)
            for k in range(0, len(test_data[new_col])):
                test_data[new_col][k] = test_data[col_name][k]

    test_data.to_csv("y.csv", sep=',')

if __name__=="__main__":
    preprocess("training.csv")
    convertTest("test.csv")
