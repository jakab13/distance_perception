import numpy as np
import slab
import os
import pathlib
import pandas as pd
from csv import writer
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DIR = pathlib.Path(os.getcwd())

folder_path = DIR / 'experiment' / 'results'

file_names = [os.path.join(path, name) for path, subdirs, files in os.walk(folder_path) for name in files]

df = pd.DataFrame({'file_name' : [],
                   'subject_ID' : [],
                   'duration' : [],
                   'correct_in_%' : [],
                   'mse' : [],
                   'slope' : []})
threshold_file = df.to_csv(DIR / 'experiment' / 'analysis' / 'results.csv')

for f in file_names:
    subject_ID = slab.ResultsFile.read_file(f, tag='subject_ID')
    duration = slab.ResultsFile.read_file(f, tag='duration')
    seq_data = slab.ResultsFile.read_file(f, tag='trial')
    seq_data = seq_data["data"]
    correct_answer = [int(i[2]) for i in seq_data]
    correct = (sum(correct_answer)) / len(correct_answer)
    expected = [int(i[0]) for i in seq_data]
    answer = [int(i[1]) for i in seq_data]
    mse = (sum(np.square(np.subtract(expected, answer)))) / len(expected)
    slope, intercept, r, p, std_err = stats.linregress(expected, answer)
    def linreg(x):
        return slope * x + intercept
    list = [[], f, subject_ID, duration, correct, mse, slope] #do list_1 here so it doesnt interfere with linreg
    with open(DIR / 'experiment' / 'analysis' / 'results.csv', 'a', newline = '') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list)
        f_object.close()
print("finished")

