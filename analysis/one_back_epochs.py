import json
import pathlib
import os

DIR = pathlib.Path(os.getcwd())
folder_path = DIR / 'analysis' / 'data' / 'USOs'
input_file_path = folder_path / 'uso_ids_and_conditions_new_new.json'
output_file_path = folder_path / 'one_back_uso_epochs_new_new.json'

# Opening JSON file
with open(input_file_path) as json_file:
    data = json.load(json_file)

output = dict()
n_conditions = 6

for participant, epochs in data.items():
    one_back_matrix = [[list() for x in range(n_conditions)] for y in range(n_conditions)]
    for idx, epoch in enumerate(epochs):
        current_condition = epoch["condition"]
        if current_condition is not None:
            one_back_idx = idx - 1
            one_back_condition = epochs[one_back_idx]["condition"]
            if one_back_condition is not None:
                one_back_matrix[current_condition][one_back_condition].append(idx)
    output.update({participant: one_back_matrix})

with open(output_file_path, "w") as outfile:
    json.dump(output, outfile)
