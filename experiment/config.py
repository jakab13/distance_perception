import pathlib
import numpy

DIR = pathlib.Path(__file__).parent.absolute()


def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
    distance_groups = {
        'linear_10': {
            1: [20],
            2: [40],
            3: [80],
            4: [160],
            5: [320],
            6: [640],
            7: [960],
            8: [1280],
            9: [1920]
        },
        'linear_5': {
            1: numpy.arange(20, 80, 20),
            2: numpy.arange(500, 580, 20),
            3: numpy.arange(980, 1060, 20),
            4: numpy.arange(1460, 1540, 20),
            5: numpy.arange(1940, 2020, 20)
        },
        'log_10': {
            1: numpy.arange(20, 60, 20),
            2: numpy.arange(140, 180, 20),
            3: numpy.arange(380, 440, 20),
            4: numpy.arange(560, 640, 20),
            5: numpy.arange(780, 860, 20),
            6: numpy.arange(980, 1020, 20),
            7: numpy.arange(1180, 1280, 20),
            8: numpy.arange(1580, 1660, 20),
            9: numpy.arange(2200, 2300, 20),
            10: numpy.arange(2900, 3000, 20)
        },
        'log_5': {
            1: numpy.arange(20, 60, 20),
            2: numpy.arange(380, 440, 20),
            3: numpy.arange(780, 860, 20),
            4: numpy.arange(1580, 1660, 20),
            5: numpy.arange(2900, 3000, 20)
        },
        'log_5_full': {
            1: numpy.arange(20, 60, 20),
            2: numpy.arange(200, 280, 20),
            3: numpy.arange(780, 860, 20),
            4: numpy.arange(1360, 1440, 20),
            5: numpy.arange(1800, 2000, 20)
        },
        'vocal_effort': {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5]
        },
        'USO_scale': {
            1: [17, 18],
            2: [57, 60],
            3: [190, 201],
            4: [636, 673],
            5: [2125, 2250]
        }
    }
    selected_USO_IDs = [2, 3, 5, 7, 10, 11, 14, 17, 18, 19, 20, 21, 22, 25, 27, 28]
    config_obj['proc_list'] = proc_list
    config_obj['distance_groups'] = distance_groups
    config_obj['selected_USO_IDs'] = selected_USO_IDs
    return config_obj
