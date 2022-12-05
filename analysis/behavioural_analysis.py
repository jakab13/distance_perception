import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

experiment = "vocal_effort"  # either "noise" or "laughter"
# get pilot folder directory.
DIR = pathlib.Path(os.getcwd())
sub_DIR = DIR / "analysis" / "data" / f"{experiment}"

ids = list(name for name in os.listdir(sub_DIR) if os.path.isdir(os.path.join(sub_DIR, name)))
behaviours = {id: {} for id in ids}

for id in ids:
    test_file_names = [f for f in listdir(sub_DIR / id)
                      if isfile(join(sub_DIR / id, f))
                      and not f.startswith('.')
                      and 'test' in f]
    training_file_names = [f for f in listdir(sub_DIR / id)
                       if isfile(join(sub_DIR / id, f))
                       and not f.startswith('.')
                       and 'training' in f]
    scores = []
    for file_name in test_file_names:
        correct_total = slab.ResultsFile.read_file(sub_DIR / id / file_name, tag="correct_total")
        scores.append(correct_total[-1])
    training_length = 0
    for training_file_name in training_file_names:
        sequence = slab.ResultsFile.read_file(sub_DIR / id / training_file_name, tag="sequence")
        training_length += sequence["n_trials"]

    average_score = sum(scores)/len(scores)
    behaviours[id] = {"average_score": average_score,
                      "training_length": training_length}

df = pd.DataFrame.from_dict(behaviours)
dft = df.transpose()
dft.to_csv(r'behaviours.csv', header=True)