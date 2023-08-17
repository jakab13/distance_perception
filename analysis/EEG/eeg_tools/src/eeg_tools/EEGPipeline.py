from eeg_tools import utils

class Pipeline:
    def __init__(self, root_dir=None):
        data_dir = None
        raws = None
        epochs = None
        evokeds = None
        if root_dir:
            mapping = None