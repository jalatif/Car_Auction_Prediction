__author__ = 'manshu'

import pandas as pd
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("credit-data-trainingset.csv")

print df.head
features = np.array(['revolving_utilization_of_unsecured_lines',
                     'age', 'number_of_time30-59_days_past_due_not_worse',
                     'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans',
                     'number_of_times90_days_late', 'number_real_estate_loans_or_lines',
                     'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents'])

clf = RandomForestClassifier()#compute_importances=True
clf.fit(df[features], df['serious_dlqin2yrs'])



# from the calculated importances, order them from most to least important
# and make a barplot so we can visualize what is/isn't important
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, features[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")
pl.show()

df['income_bins'] = pd.cut(df.monthly_income, bins=15)
pd.value_counts(df['income_bins'])
# not very helpful

print "Monar"
print df.monthly_income

def cap_values(x, cap):
    if x > cap:
        return cap
    else:
        return x

df.monthly_income = df.monthly_income.apply(lambda x: cap_values(x, 15000))

print df.monthly_income.describe()

df['income_bins'] = pd.cut(df.monthly_income, bins=15, labels=False)
pd.value_counts(df.income_bins)
