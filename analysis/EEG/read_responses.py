import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pathlib
import os
import slab
import seaborn as sns
from scipy.stats import linregress

# TODO: Fix seqs dict() so that it containts all the information properly.

DIR = pathlib.Path(os.getcwd())
experiment = "vocal_effort"  # "vocal_effort" or "noise" data.
data_DIR = DIR / "analysis" / "data" / f"{experiment}"
ids = list(name for name in os.listdir(data_DIR)
           if os.path.isdir(os.path.join(data_DIR, name)))
subject = ids[0]
file_type = ".txt"
_results_folder = data_DIR / subject
slab.ResultsFile.results_folder = _results_folder
seqs = dict()

for root, dirs, files in os.walk(_results_folder):
    for i, file in enumerate(files):
        if file.startswith(subject) and file.endswith(file_type):
            seqs[str(i)] = slab.ResultsFile.read_file(filename=root+'/'+str(file), tag="sequence")
solutions = []
responses = []
rts = []
correct_totals = []
is_corrects = []
for list in seqs["2"]["data"]:
    for dict in list:
        solutions.append(dict["solution"])
        responses.append(dict["response"])
        is_corrects.append(dict["isCorrect"])
        correct_totals.append(dict["correct_total"])
        rts.append(dict["rt"])

df = pd.DataFrame(columns=dict.keys())
df.solution, df.response, df.rt, df.isCorrect, df.correct_total = solutions, responses, rts, is_corrects, correct_totals
correct_perc = df.correct_total.max() / seqs["2"]["this_n"]
slope, intercept, rv, pv, stderr = linregress(df.solution, df.response)
sns.regplot(x=df.solution, y=df.response, data=df)
plt.title(f"correctness percentage: {correct_perc}, distance gain: {slope:.2f}")
