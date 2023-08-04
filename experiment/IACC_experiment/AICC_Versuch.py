import random
import slab
import os
import pathlib
from os.path import join
import csv
import re
import matplotlib
import math
folder_path = pathlib.Path
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import slab
import pathlib
import os
from os import listdir
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
DIR = pathlib.Path(os.getcwd())

# reading in files for analysis

###############################################

# Result files Hannah

Hannah_02_01 = slab.ResultsFile.read_file(DIR / 'IACC_experiment' / 'Results' / 'stevens_scaling' / 'Hannah_2023-03-28-10-52-09.txt')
Hannah_02_02 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Hannah/Hannah_2023-03-28-11-16-58.txt')
Hannah_02_03 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Hannah/Hannah_2023-03-28-11-22-56.txt')


################ Hannah_02_01 #####################

Hannah_02_01_orig = list()  # _orig is the correct answers (the predicted outcome)
IACC_vals = list()
Hannah_02_01_responses  = list()  # _responses is the actual value. so the response given by the participants

for trial in Hannah_02_01['IACC_distance']['data']: #Here we are generating a new list containing the responses
        Hannah_02_01_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Hannah_02_01['IACC_distance']['data']: #Here we are generating a new list containing the predicted values
        Hannah_02_01_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Hannah_02_01_responses_int = [int(val) for val in Hannah_02_01_responses] #Because the responses are saved as strings
#we have to turn them into integers first.

Hannah_02_01_CM = confusion_matrix(Hannah_02_01_responses_int, Hannah_02_01_orig) #generating a confusion matrix (as an array)
Hannah_02_01_final = pd.DataFrame(Hannah_02_01_CM, index = [i for i in "12345"], #adding labels for the rows and columns
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7)) #setting the size of the confusion matrix
sn.heatmap(Hannah_02_01_final, annot=True) #plotting the matrix with the correct values

# Accuracy score and mean squared error

acc_Hannah_02_01 = round(100 * metrics.accuracy_score(Hannah_02_01_responses_int, Hannah_02_01_orig), 2)
print(acc_Hannah_02_01)
mean_Hannah_02_01 = round(metrics.mean_squared_error(Hannah_02_01_responses_int, Hannah_02_01_orig), 2)
print(mean_Hannah_02_01)

################ Hannah_02_02 #####################

Hannah_02_02_orig = list()
IACC_vals = list()
Hannah_02_02_responses  = list()

for trial in Hannah_02_02['IACC_distance']['data']:
        Hannah_02_02_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Hannah_02_02['IACC_distance']['data']:
        Hannah_02_02_orig.append(int(trial[0]['distance_group']))


# plotting confusion Matrix

Hannah_02_02_responses_int = [int(val) for val in Hannah_02_02_responses]

Hannah_02_02_CM = confusion_matrix(Hannah_02_02_responses_int, Hannah_02_02_orig)
Hannah_02_02_final = pd.DataFrame(Hannah_02_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Hannah_02_02_final, annot=True)

# Accuracy score and mean squared error

acc_Hannah_02_02 = round(100 * metrics.accuracy_score(Hannah_02_02_responses_int, Hannah_02_02_orig), 2)
print(acc_Hannah_02_02)
mean_Hannah_02_02 = round(metrics.mean_squared_error(Hannah_02_02_responses_int, Hannah_02_02_orig), 2)
print(mean_Hannah_02_02)

################ Hannah_02_03 #####################

Hannah_02_03_orig = list()
IACC_vals = list()
Hannah_02_03_responses  = list()

for trial in Hannah_02_03['IACC_distance']['data']:
        Hannah_02_03_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Hannah_02_03['IACC_distance']['data']:
        Hannah_02_03_orig.append(int(trial[0]['distance_group']))


# plotting confusion Matrix

Hannah_02_03_responses_int = [int(val) for val in Hannah_02_03_responses]

Hannah_02_03_CM = confusion_matrix(Hannah_02_03_responses_int, Hannah_02_03_orig)
Hannah_02_03_final = pd.DataFrame(Hannah_02_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Hannah_02_03_final, annot=True)

# Accuracy score and mean squared error

acc_Hannah_02_03 = round(100 * metrics.accuracy_score(Hannah_02_03_responses_int, Hannah_02_03_orig), 2)
print(acc_Hannah_02_03)
mean_Hannah_02_03 = round(metrics.mean_squared_error(Hannah_02_03_responses_int, Hannah_02_03_orig), 2)
print(mean_Hannah_02_03)

###################################################
###################################################

# Result files Marius

Marius_28_01 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Marius_28/Marius_28_2023-03-28-15-27-52.txt')
Marius_28_02 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Marius_28/Marius_28_2023-03-28-15-39-58.txt')
Marius_28_03 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Marius_28/Marius_28_2023-03-28-15-48-07.txt')


################# Marius_28_01 ####################

Marius_28_01_orig = list()
IACC_vals = list()
Marius_28_01_responses  = list()

for trial in Marius_28_01['IACC_distance']['data']:
        Marius_28_01_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Marius_28_01['IACC_distance']['data']:
        Marius_28_01_orig.append(int(trial[0]['distance_group']))


# plotting confusion Matrix

Marius_28_01_responses_int = [int(val) for val in Marius_28_01_responses]

Marius_28_01_CM = confusion_matrix(Marius_28_01_responses_int, Marius_28_01_orig)
Marius_28_01_final = pd.DataFrame(Marius_28_01_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Marius_28_01_final, annot=True)

# Accuracy score and mean squared error

acc_Marius_28_01 = round(100 * metrics.accuracy_score(Marius_28_01_responses_int, Marius_28_01_orig), 2)
print(acc_Marius_28_01)
mean_Marius_28_01 = round(metrics.mean_squared_error(Marius_28_01_responses_int, Marius_28_01_orig), 2)
print(mean_Marius_28_01)

################ Marius_28_02 ####################

Marius_28_02_orig = list()
IACC_vals = list()
Marius_28_02_responses  = list()

for trial in Marius_28_02['IACC_distance']['data']:
        Marius_28_02_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Marius_28_02['IACC_distance']['data']:
        Marius_28_02_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Marius_28_02_responses_int = [int(val) for val in Marius_28_02_responses]

Marius_28_02_CM = confusion_matrix(Marius_28_02_responses_int, Marius_28_02_orig)
Marius_28_02_final = pd.DataFrame(Marius_28_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Marius_28_02_final, annot=True)

# Accuracy score and mean squared error

acc_Marius_28_02 = round(100 * metrics.accuracy_score(Marius_28_02_responses_int, Marius_28_02_orig), 2)
print(acc_Marius_28_02)
mean_Marius_28_02 = round(metrics.mean_squared_error(Marius_28_02_responses_int, Marius_28_02_orig), 2)
print(mean_Marius_28_02)

################ Marius_28_03 #####################

Marius_28_03_orig = list()
IACC_vals = list()
Marius_28_03_responses  = list()

for trial in Marius_28_03['IACC_distance']['data']:
        Marius_28_03_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Marius_28_03['IACC_distance']['data']:
        Marius_28_03_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Marius_28_03_responses_int = [int(val) for val in Marius_28_03_responses]

Marius_28_03_CM = confusion_matrix(Marius_28_03_responses_int, Marius_28_03_orig)
Marius_28_03_final = pd.DataFrame(Marius_28_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Marius_28_03_final, annot=True)

# Accuracy score and mean squared error

acc_Marius_28_03 = round(100 * metrics.accuracy_score(Marius_28_03_responses_int, Marius_28_03_orig), 2)
print(acc_Marius_28_03)
mean_Marius_28_03 = round(metrics.mean_squared_error(Marius_28_03_responses_int, Marius_28_03_orig), 2)
print(mean_Marius_28_03)

"""""
for trial in Marius_28_01['IACC_distance']['data']:
        print(trial)

for trial in Marius_28_01['IACC_distance']['data']:
        print(trial[0]['response'])


for trial in Marius_28_01['IACC_distance']['data']:
        print(int(int(trial[0]['response'])/2))

for trial in Marius_28_01['IACC_distance']['data']:
        print(math.ceil(int(trial[0]['response'])/2))

for trial in Marius_28_01['IACC_distance']['data']:
        responses.append(math.ceil(int(trial[0]['response'])/2))
        
len(responses)
print(responses)

for trial in Marius_28_01['IACC_distance']['data']:
        print(int(trial[0]['distance_group']))

"""
###################################################
###################################################

# Result files Kirke

Kirke_02_01 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Kirke/Kirke_2023-03-28-11-59-54.txt')
Kirke_02_02 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Kirke/Kirke_2023-03-28-12-03-26.txt')
Kirke_02_03 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Kirke/Kirke_2023-03-28-13-11-23.txt')

################ Kirke_02_01 #####################

Kirke_02_01_orig = list()
IACC_vals = list()
Kirke_02_01_responses  = list()

for trial in Kirke_02_01['IACC_distance']['data']:
        Kirke_02_01_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Kirke_02_01['IACC_distance']['data']:
        Kirke_02_01_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Kirke_02_01_responses_int = [int(val) for val in Kirke_02_01_responses]

Kirke_02_01_CM = confusion_matrix(Kirke_02_01_responses_int, Kirke_02_01_orig)
Kirke_02_01_final = pd.DataFrame(Kirke_02_01_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Kirke_02_01_final, annot=True)

# Accuracy score and mean squared error

acc_Kirke_02_01 = round(100 * metrics.accuracy_score(Kirke_02_01_responses_int, Kirke_02_01_orig), 2)
print(acc_Kirke_02_01)
mean_Kirke_02_01 = round(metrics.mean_squared_error(Kirke_02_01_responses_int, Kirke_02_01_orig), 2)
print(mean_Kirke_02_01)

################ Kirke_02_02 #####################

Kirke_02_02_orig = list()
IACC_vals = list()
Kirke_02_02_responses  = list()

for trial in Kirke_02_02['IACC_distance']['data']:
        Kirke_02_02_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Kirke_02_02['IACC_distance']['data']:
        Kirke_02_02_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Kirke_02_02_responses_int = [int(val) for val in Kirke_02_02_responses]

Kirke_02_02_CM = confusion_matrix(Kirke_02_02_responses_int, Kirke_02_02_orig)
Kirke_02_02_final = pd.DataFrame(Kirke_02_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Kirke_02_02_final, annot=True)

# Accuracy score and mean squared error

acc_Kirke_02_02 = round(100 * metrics.accuracy_score(Kirke_02_02_responses_int, Kirke_02_02_orig), 2)
print(acc_Kirke_02_02)
mean_Kirke_02_02 = round(metrics.mean_squared_error(Kirke_02_02_responses_int, Kirke_02_02_orig), 2)
print(mean_Kirke_02_02)

################ Kirke_02_03 #####################

Kirke_02_03_orig = list()
IACC_vals = list()
Kirke_02_03_responses  = list()

for trial in Kirke_02_03['IACC_distance']['data']:
        Kirke_02_03_responses.append(math.ceil(int(trial[0]['response'])/2))

for trial in Kirke_02_03['IACC_distance']['data']:
        Kirke_02_03_orig.append(int(trial[0]['distance_group']))


# plotting confusion Matrix

Kirke_02_03_responses_int = [int(val) for val in Kirke_02_03_responses]

Kirke_02_03_CM = confusion_matrix(Kirke_02_03_responses_int, Kirke_02_03_orig)
Kirke_02_03_final = pd.DataFrame(Kirke_02_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Kirke_02_03_final, annot=True)

# Accuracy score and mean squared error

acc_Kirke_02_03 = round(100 * metrics.accuracy_score(Kirke_02_03_responses_int, Kirke_02_03_orig), 2)
print(acc_Kirke_02_03)
mean_Kirke_02_03 = round(metrics.mean_squared_error(Kirke_02_03_responses_int, Kirke_02_03_orig), 2)
print(mean_Kirke_02_03)

#####################################################
#####################################################

# Result files Jakab

Jakab_17_01 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Jakab_17/Jakab_17_2023-03-28-13-45-16.txt')
Jakab_17_02 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Jakab_17/Jakab_17_2023-03-28-13-53-26.txt')
Jakab_17_03 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Jakab_17/Jakab_17_2023-03-28-13-56-41.txt')

################ Jakab_17_01 #####################

Jakab_17_01_orig = list()
IACC_vals = list()
Jakab_17_01_responses  = list()

for trial in Jakab_17_01['IACC_distance']['data']:
        Jakab_17_01_responses.append(int(trial[0]['response']))

for trial in Jakab_17_01['IACC_distance']['data']:
        Jakab_17_01_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Jakab_17_01_responses_int = [int(val) for val in Jakab_17_01_responses]

Jakab_17_01_CM = confusion_matrix(Jakab_17_01_responses_int, Jakab_17_01_orig)
Jakab_17_01_final = pd.DataFrame(Jakab_17_01_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Jakab_17_01_final, annot=True)

# Accuracy score and mean squared error

acc_Jakab_17_01 = round(100 * metrics.accuracy_score(Jakab_17_01_responses_int, Jakab_17_01_orig), 2)
print(acc_Jakab_17_01)
mean_Jakab_17_01 = round(metrics.mean_squared_error(Jakab_17_01_responses_int, Jakab_17_01_orig), 2)
print(mean_Jakab_17_01)

################ Jakab_17_02 #####################

Jakab_17_02_orig = list()
IACC_vals = list()
Jakab_17_02_responses  = list()

for trial in Jakab_17_02['IACC_distance']['data']:
        Jakab_17_02_responses.append(int(trial[0]['response']))

for trial in Jakab_17_02['IACC_distance']['data']:
        Jakab_17_02_orig.append(int(trial[0]['distance_group']))

# plotting confusion Matrix

Jakab_17_02_responses_int = [int(val) for val in Jakab_17_02_responses]

Jakab_17_02_CM = confusion_matrix(Jakab_17_02_responses_int, Jakab_17_02_orig)
Jakab_17_02_final = pd.DataFrame(Jakab_17_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Jakab_17_02_final, annot=True)

# Accuracy score and mean squared error

acc_Jakab_17_02 = round(100 * metrics.accuracy_score(Jakab_17_02_responses_int, Jakab_17_02_orig), 2)
print(acc_Jakab_17_02)
mean_Jakab_17_02 = round(metrics.mean_squared_error(Jakab_17_02_responses_int, Jakab_17_02_orig), 2)
print(mean_Jakab_17_02)

################ Jakab_17_03 #####################

Jakab_17_03_orig = list()
IACC_vals = list()
Jakab_17_03_responses  = list()

for trial in Jakab_17_03['IACC_distance']['data']:
        Jakab_17_03_responses.append(int(trial[0]['response']))

for trial in Jakab_17_03['IACC_distance']['data']:
        Jakab_17_03_orig.append(int(trial[0]['distance_group']))


# plotting confusion matrix

Jakab_17_03_responses_int = [int(val) for val in Jakab_17_03_responses]

Jakab_17_03_CM = confusion_matrix(Jakab_17_03_responses_int, Jakab_17_03_orig)
Jakab_17_03_final = pd.DataFrame(Jakab_17_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Jakab_17_03_final, annot=True)

# Accuracy score and mean squared error

acc_Jakab_17_03 = round(100 * metrics.accuracy_score(Jakab_17_03_responses_int, Jakab_17_03_orig), 2)
print(acc_Jakab_17_03)
mean_Jakab_17_03 = round(metrics.mean_squared_error(Jakab_17_03_responses_int, Jakab_17_03_orig), 2)
print(mean_Jakab_17_03)

#############################################################
#############################################################

# Result files Luca

Luca_10_01 = slab.ResultsFile.read_file('/Users/jakabpilaszanovich/My Drive/PhD?/Leipzig/Free Field Lab/P1 - Distance Perception/Behavioural Experiments/IACC/Results/Luca_10/Luca_10_2023-03-31-08-37-49.txt')
Luca_10_02 = slab.ResultsFile.read_file('/Users/jakabpilaszanovich/My Drive/PhD?/Leipzig/Free Field Lab/P1 - Distance Perception/Behavioural Experiments/IACC/Results/Luca_10/Luca_10_2023-03-31-08-41-59.txt')
Luca_10_03 = slab.ResultsFile.read_file('/Users/jakabpilaszanovich/My Drive/PhD?/Leipzig/Free Field Lab/P1 - Distance Perception/Behavioural Experiments/IACC/Results/Luca_10/Luca_10_2023-03-31-08-45-27.txt')

################ Luca_10_01 #####################

Luca_10_01_orig = list()
IACC_vals = list()
Luca_10_01_responses  = list()

for trial in Luca_10_01['IACC_distance']['data']:
        Luca_10_01_responses.append(int(trial[0]['response']))

for trial in Luca_10_01['IACC_distance']['data']:
        Luca_10_01_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Luca_10_01_responses_int = [int(val) for val in Luca_10_01_responses]

Luca_10_01_CM = confusion_matrix(Luca_10_01_responses_int, Luca_10_01_orig)
Luca_10_01_final = pd.DataFrame(Luca_10_01_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Luca_10_01_final, annot=True)
plt.savefig("IACC_CM_sub_10_before.eps", format="eps")

# Accuracy score and mean squared error

acc_Luca_10_01 = round(100 * metrics.accuracy_score(Luca_10_01_responses_int, Luca_10_01_orig), 2)
print(acc_Luca_10_01)
mean_Luca_10_01 = round(metrics.mean_squared_error(Luca_10_01_responses_int, Luca_10_01_orig), 2)
print(mean_Luca_10_01)

################ Luca_10_02 #####################

Luca_10_02_orig = list()
IACC_vals = list()
Luca_10_02_responses  = list()

for trial in Luca_10_02['IACC_distance']['data']:
        Luca_10_02_responses.append(int(trial[0]['response']))

for trial in Luca_10_02['IACC_distance']['data']:
        Luca_10_02_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Luca_10_02_responses_int = [int(val) for val in Luca_10_02_responses]

Luca_10_02_CM = confusion_matrix(Luca_10_02_responses_int, Luca_10_02_orig)
Luca_10_02_final = pd.DataFrame(Luca_10_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Luca_10_02_final, annot=True)

# Accuracy score and mean squared error

acc_Luca_10_02 = round(100 * metrics.accuracy_score(Luca_10_02_responses_int, Luca_10_02_orig), 2)
print(acc_Luca_10_02)
mean_Luca_10_02 = round(metrics.mean_squared_error(Luca_10_02_responses_int, Luca_10_02_orig), 2)
print(mean_Luca_10_02)

################ Luca_10_03 #####################

Luca_10_03_orig = list()
IACC_vals = list()
Luca_10_03_responses  = list()

for trial in Luca_10_03['IACC_distance']['data']:
        Luca_10_03_responses.append(int(trial[0]['response']))

for trial in Luca_10_03['IACC_distance']['data']:
        Luca_10_03_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Luca_10_03_responses_int = [int(val) for val in Luca_10_03_responses]

Luca_10_03_CM = confusion_matrix(Luca_10_03_responses_int, Luca_10_03_orig)
Luca_10_03_final = pd.DataFrame(Luca_10_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Luca_10_03_final, annot=True)
plt.savefig("IACC_CM_sub_10_after.eps", format="eps")

# Accuracy score and mean squared error

acc_Luca_10_03 = round(100 * metrics.accuracy_score(Luca_10_03_responses_int, Luca_10_03_orig), 2)
print(acc_Luca_10_03)
mean_Luca_10_03 = round(metrics.mean_squared_error(Luca_10_03_responses_int, Luca_10_03_orig), 2)
print(mean_Luca_10_03)

#############################################################
#############################################################

# Result files Carsten
Carsten_22_01 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Carsten_22/Carsten_22_2023-03-31-14-05-55.txt')
Carsten_22_02 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Carsten_22/Carsten_22_2023-03-31-14-17-32.txt')
Carsten_22_03 = slab.ResultsFile.read_file('/Users/kirkekley/PycharmProjects/pythonProject/Results/Carsten_22/Carsten_22_2023-03-31-14-22-25.txt')

################ Carsten_22_01 #####################

Carsten_22_01_orig = list()
IACC_vals = list()
Carsten_22_01_responses  = list()

for trial in Carsten_22_01['IACC_distance']['data']:
        Carsten_22_01_responses.append(int(trial[0]['response']))

for trial in Carsten_22_01['IACC_distance']['data']:
        Carsten_22_01_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Carsten_22_01_responses_int = [int(val) for val in Carsten_22_01_responses]

Carsten_22_01_CM = confusion_matrix(Carsten_22_01_responses_int, Carsten_22_01_orig)
Carsten_22_01_final = pd.DataFrame(Carsten_22_01_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Carsten_22_01_final, annot=True)

# Accuracy score and mean squared error

acc_Carsten_22_01 = round(100 * metrics.accuracy_score(Carsten_22_01_responses_int, Carsten_22_01_orig), 2)
print(acc_Carsten_22_01)
mean_Carsten_22_01 = round(metrics.mean_squared_error(Carsten_22_01_responses_int, Carsten_22_01_orig), 2)
print(mean_Carsten_22_01)

################ Carsten_22_02 #####################

Carsten_22_02_orig = list()
IACC_vals = list()
Carsten_22_02_responses  = list()

for trial in Carsten_22_02['IACC_distance']['data']:
        Carsten_22_02_responses.append(int(trial[0]['response']))

for trial in Carsten_22_02['IACC_distance']['data']:
        Carsten_22_02_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Carsten_22_02_responses_int = [int(val) for val in Carsten_22_02_responses]

Carsten_22_02_CM = confusion_matrix(Carsten_22_02_responses_int, Carsten_22_02_orig)
Carsten_22_02_final = pd.DataFrame(Carsten_22_02_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Carsten_22_02_final, annot=True)

# Accuracy score and mean squared error

acc_Carsten_22_02 = round(100 * metrics.accuracy_score(Carsten_22_02_responses_int, Carsten_22_02_orig), 2)
print(acc_Carsten_22_02)
mean_Carsten_22_02 = round(metrics.mean_squared_error(Carsten_22_02_responses_int, Carsten_22_02_orig), 2)
print(mean_Carsten_22_02)


################ Carsten_22_03 #####################

Carsten_22_03_orig = list()
IACC_vals = list()
Carsten_22_03_responses  = list()

for trial in Carsten_22_03['IACC_distance']['data']:
        Carsten_22_03_responses.append(int(trial[0]['response']))

for trial in Carsten_22_03['IACC_distance']['data']:
        Carsten_22_03_orig.append(int(trial[0]['distance_group']))

# plotting confusion matrix

Carsten_22_03_responses_int = [int(val) for val in Carsten_22_03_responses]

Carsten_22_03_CM = confusion_matrix(Carsten_22_03_responses_int, Carsten_22_03_orig)
Carsten_22_03_final = pd.DataFrame(Carsten_22_03_CM, index = [i for i in "12345"],
                   columns = [i for i in "12345"])
plt.figure(figsize = (10,7))
sn.heatmap(Carsten_22_03_final, annot=True)

# Accuracy score and mean squared error

acc_Carsten_22_03 = round(100 * metrics.accuracy_score(Carsten_22_03_responses_int, Carsten_22_03_orig), 2)
print(acc_Carsten_22_03)
mean_Carsten_22_03 = round(metrics.mean_squared_error(Carsten_22_03_responses_int, Carsten_22_03_orig), 2)
print(mean_Carsten_22_03)

