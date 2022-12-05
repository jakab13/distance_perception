import sys
from pathlib import Path
from analysis.utils import get_epochs, apply_transform
import argparse
import numpy as np
from meegkit.dss import dss0, dss1
from meegkit.utils.covariances import tscov
root = Path(__file__).parent.parent.absolute()
sys.path.append(str(root/"code"))


def compute_transformation(epochs, condition1, condition2, keep):

    if not (condition1 in epochs.events[:, 2] and condition2 in epochs.events[:, 2]):
        raise ValueError("'conditions' must be values of two event types!")
    X = epochs.get_data().transpose(2, 1, 0)
    events = epochs.events

    to_jd1, from_jd1, _, pwr = dss1(X)  # compute the transformations
    del X
    to_jd1 = to_jd1[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd1 = from_jd1[np.argsort(pwr)[::-1], :]
    to_jd1 = to_jd1[:, 0:keep]  # only keep the largest ones
    from_jd1 = from_jd1[0:keep, :]

    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components

    idx1 = np.where(events[:, 2] == condition1)[0]
    idx2 = np.where(events[:, 2] == condition2)[0]
    D = Y[idx1, :, :].mean(axis=0) - Y[idx2, :, :].mean(axis=0)    # compute the difference between conditions
    Y, D = Y.T, D.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, _, pwr = dss0(c0, c1)  # compute the transformations
    to_jd2 = to_jd2[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd2 = from_jd2[np.argsort(pwr)[::-1], :]

    return to_jd1, from_jd1, to_jd2, from_jd2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Use joint decorrelation to compute unmixing matrices with bootstrapping.")
    parser.add_argument('experiment', type=str, choices=["freefield", "headphones"])
    parser.add_argument('subject', type=str)
    parser.add_argument('keep', type=int, help="number of components to keep after the first jd.")
    parser.add_argument('conditions', type=int, nargs=2, help="ids of conditions for difference.")
    args = parser.parse_args()

    out_folder = root / args.experiment / "input" / "transforms" / args.subject
    if not out_folder.exists():
        out_folder.mkdir()
    epochs = get_epochs(args.subject)
    epochs = epochs.equalize_event_counts(epochs.event_id.keys())[1]
    to_jd1, from_jd1, to_jd2, from_jd2 = \
        compute_transformation(epochs, args.conditions[0], args.conditions[1], args.keep)
    for matrix, name in zip([to_jd1, to_jd2, from_jd1, from_jd2], ["to_jd1", "to_jd2", "from_jd1", "from_jd2"]):
        np.save(out_folder/name, matrix)
