import pathlib
import os

DIR = pathlib.Path(os.getcwd())

def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2',  DIR / 'experiment' / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8',  DIR / 'experiment' / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'experiment' / 'data' / 'play_buf.rcx']]
    config_obj['proc_list'] = proc_list
    return config_obj