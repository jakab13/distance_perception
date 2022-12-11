import json
import pathlib
import os

DIR = pathlib.Path(os.getcwd())
folder_path = DIR / 'analysis' / 'data' / 'USOs'
input_file_path = 'uso_ids_and_conditions.json'
output_file_path = folder_path / 'one_back_uso_epochs.json'

# Opening JSON file
with open(input_file_path) as json_file:
    data = json.load(json_file)

output = dict()
n_conditions = 6

for participant, epochs in data.items():
    one_back_matrix = [[list() for x in range(n_conditions)] for y in range(n_conditions)]
    for idx, epoch in enumerate(epochs):
        if idx > 0:  # Don't count first epoch
            condition = epoch["condition"]
            one_back_idx = idx - 1
            one_back_condition = epochs[one_back_idx]["condition"]
            one_back_matrix[condition][one_back_condition].append(one_back_idx)
    output.update({participant: one_back_matrix})

with open(output_file_path, "w") as outfile:
    json.dump(output, outfile)