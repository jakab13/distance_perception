import os
import pathlib
import re
import json
import mne
import fnmatch

# TODO: all function need root_dir. Find a way to avoid that.


def find(path, mode="pattern", pattern=None, string=None):
    """
    Mode can be "pattern" or "name".
    """
    found_files = []
    if mode == "string":
        for root, dirs, files in os.walk(path):
            if string in files:
                found_files.append(pathlib.Path(os.path.join(root, string)))
    if mode == "pattern":
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    found_files.append(pathlib.Path(os.path.join(root, name)))
    if len(found_files) == 1:
        return found_files[0]
    else:
        return found_files


def load_file(dir, type="mapping", format="json"):
    if type == "montage":
        montage_path = find(path=dir, mode="pattern", pattern="*.bvef")
        montage = mne.channels.read_custom_montage(fname=montage_path)
        return montage
    elif type == "ica":
        ica_path = find(path=dir, mode="pattern", pattern="*ica.fif")
        ica_ref = mne.preprocessing.read_ica(ica_path)
        return ica_ref
    elif type == "config":
        config_path = find(path=dir, mode="pattern", pattern="*config.json")
        with open(config_path) as file:
            lf = json.load(file)
        return lf
    else:
        fp = find(path=dir, mode="pattern", pattern=f"*{type}.{format}")
        with open(fp) as file:
            lf = json.load(file)
        return lf


# r"" == raw string
# \b matches on a change from a \w (a word character) to a \W (non word character)
# \w{6} == six alphanumerical characters
# RegEx expression to match subject ids (6 alphanumerical characters)
def get_ids_from_header_files(header_files, id_chars):
    ids = []
    regexp = r'\b\w{%s}\b' % (id_chars)
    for header_file in header_files:
        match = re.search(pattern=regexp, string=header_file)
        if match.group() not in ids:
            ids.append(match.group())
    return ids


def get_ids_from_data_dir(data_dir):
    ids = list(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)))
    return ids


def generate_folders(root_dir, id=None, folders=["epochs", "raw", "evokeds", "figures"]):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            for folder in folders:
                folder_path = pathlib.Path(root) / folder
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)
                    print(f"{folder} folder in {id} successfully generated!")
                else:
                    print(f"{folder} folder already exists in {id}!")
        else:
            continue


def save_object(data, root_dir, id, overwrite=True):
    if isinstance(data, mne.io.brainvision.brainvision.RawBrainVision):
        folder_path = pathlib.Path(root_dir) / id / "raw"
        data.save(f"{folder_path}/{id}_raw.fif", overwrite=overwrite)
    elif isinstance(data, mne.Epochs):
        folder_path = pathlib.Path(root_dir) / id / "epochs"
        data.save(f"{folder_path}/{id}-epo.fif", overwrite=overwrite)
    elif isinstance(data, mne.Evoked) or isinstance(data, list):
        file_path = pathlib.Path(str(root_dir + "/" + id + "/evokeds/"))
        if isinstance(data, list):
            mne.write_evokeds(f"{file_path}/{id}-ave.fif", data)
        else:
            data.save(f"{file_path}/{id}-ave.fif", overwrite=overwrite)
    else:
        print("Data needs to be an mne object of type mne.io.Raw, mne.Epochs or mne.Evoked!")


def read_object(data_type, root_dir, id, condition=None):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            if data_type == "raw":
                folder_path = pathlib.Path(root) / "raw"
                raw = mne.io.read_raw_fif(f"{folder_path}\\{id}_raw.fif", preload=True)
                return raw
            if data_type == "epochs":
                folder_path = pathlib.Path(root) / "epochs"
                epochs = mne.read_epochs(f"{folder_path}\\{id}-epo.fif", preload=True)
                return epochs
            if data_type == "evokeds":
                folder_path = pathlib.Path(root) / "evokeds"
                evokeds = mne.read_evokeds(f"{folder_path}\\{id}-ave.fif", condition=condition)
                return evokeds


def check_id(id, root_dir):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            evkd_path = find(root, mode="pattern", pattern="*-ave.fif")
            if not evkd_path:
                print(f"Subject {id} has not been processed yet. Starting pipeline ...")
                return False
            elif evkd_path:
                print(f"Subject {id} has been processed already. Skipping ...")
                return True
