# import sys
import pathlib
import utils

# path = os.getcwd() + "\\src\\" + "eeg_tools"
# sys.path.append(path)

"""
This module contains necessary variables for running the preprocessing pipeline \n
on EEG data.
Callable variables are:
cfg = configuration file (dictionary) containing necessary preprocessing parameters.
ica_ref = ICA reference template.
mapping = electrode mapping according to 10-20 system.
montage = file containing the electrode coordinates on the scalp.
header_files = list containing all paths to the .vhdr files in the given directory.
ids = list of subject ids extracted from header files.
"""


# default prerequisites:
# root_dir = pathlib.Path(input("Enter path to EEG_tools main directory: "))
root_dir = "/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/eeg_tools/src/eeg_tools"
# data_dir = pathlib.Path(input("Enter path to EEG data:"))
data_dir = "/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/data/pinknoise"
try:
    cfg = utils.load_file(root_dir, type="config")
except:
    print("Config file not found. Be sure that EEG data and config file are stored in the entered path.")
try:
    ica_ref = utils.load_file(root_dir, type="ica")
except:
    print("ICA template not found. Be sure that EEG data and template file are stored the entered path.")
try:
    mapping = utils.load_file(root_dir, "mapping")
except:
    print("Electrode mapping not found. Be sure that EEG data and mapping file are stored the entered path.")
try:
    montage = utils.load_file(root_dir, "montage")
except:
    print("Electrode montage not found. Be sure that EEG data and montage file are stored the entered path.")
try:
    header_files = utils.find(path=data_dir, mode="pattern", pattern="*.vhdr")
except:
    print("VHDR files not found. Be sure that EEG data are stored the entered path.")
try:
    # id_chars = int(input("Enter subject ID character length: "))
    # ids = utils.get_ids(header_files, id_chars)
    ids = utils.get_ids_from_data_dir(data_dir)
except:
    print("IDs could not be extracted. RegEx syntax might possibly be wrong.")

# WIP
def update(id):
    fig_folder = pathlib.Path(f"D:/EEG/vocal_effort/data/{id}/figures")
    return fig_folder
