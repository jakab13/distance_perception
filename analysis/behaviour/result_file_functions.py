import pandas as pd
import slab
import pathlib
import os
from os import listdir
import mne
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json


def get_file_paths(root_path):
    participants = [p for p in listdir(root_path) if not p.startswith(".")]
    file_paths = {participant: [] for participant in participants}
    for participant in file_paths:
        folder_path = root_path + '/' + participant + '/results_files/'
        file_paths[participant] = [pathlib.Path(folder_path + f) for f in [p for p in listdir(folder_path) if not p.startswith(".")]]
    return file_paths


def get_all_results(root_path):
    file_paths = get_file_paths(root_path=root_path)
    result_files = {participant: [] for participant in file_paths}
    for participant in file_paths:
        for file in file_paths[participant]:
            result_files[participant].append(slab.ResultsFile.read_file(file))
    return result_files


def get_stage_results(stage, root_path):
    file_paths = get_file_paths(root_path=root_path)
    stage_result_files = {participant: [] for participant in file_paths}
    for participants in file_paths:
        for file in file_paths[participants]:
            if stage == 'experiment':
                is_experiment = slab.ResultsFile.read_file(file, tag='stage') == 'experiment'
                if is_experiment:
                    stage_result_files[participants].append(slab.ResultsFile.read_file(file))
            elif stage == 'test':
                is_experiment = slab.ResultsFile.read_file(file, tag='stage') == 'test'
                if is_experiment:
                    stage_result_files[participants].append(slab.ResultsFile.read_file(file))
            elif stage == 'training':
                is_experiment = slab.ResultsFile.read_file(file, tag='stage') == 'training'
                if is_experiment:
                    stage_result_files[participants].append(slab.ResultsFile.read_file(file))
    return stage_result_files


def get_number_of_tests(result_files):
    how_many_tests = {participant: 0 for participant in result_files.keys()}
    for participant in result_files:
        for file in result_files[participant]:
            if 'test' in file[0]['stage']:
                for dict in file:
                    if 'sequence' in dict:
                        how_many_tests[participant] += 1
    return how_many_tests


def get_number_of_trainings(result_files):
    how_many_trainings = {participant: 0 for participant in result_files.keys()}
    for participant in result_files:
        for file in result_files[participant]:
            if 'training' in file[0]['stage']:
                how_many_trainings[participant] += 1
    return how_many_trainings


def get_rea_times(root_path, only_concluded_tests=True):
    results = get_stage_results(stage='test', root_path=root_path)
    if only_concluded_tests:
        results = delete_not_concluded_tests(results)
    rea_times = {participant: {} for participant in results.keys()}
    for id in results:
        for test_idx, test in enumerate(results[id]):
            rea_times[id][str(test_idx)] = []
            for dicts in test:
                if 'reaction_time' in dicts and dicts['reaction_time'] is not None:
                    rea_times[id][str(test_idx)].append(dicts['reaction_time'])
    return rea_times


def get_score(root_path, result_files, only_concluded_tests=True):
    test_scores = {participant: {} for participant in result_files.keys()}
    tests = get_stage_results(stage='test', root_path=root_path)
    if only_concluded_tests:
        tests = delete_not_concluded_tests(tests)
    for participant in tests:
        for test_idx, test in enumerate(tests[participant]):
            correct_totals = list()
            for dicts in tests[participant][test_idx]:
                if 'correct_total' in dicts:
                    correct_totals.append(dicts['correct_total'])
                elif 'sequence' in dicts:
                    n = dicts['sequence']['this_n']
                    test_scores[participant][str(test_idx)] = max(correct_totals) / n
    return test_scores


def get_solutions_and_responses(root_path):
    tests = get_stage_results(stage='test', root_path=root_path)
    tests = delete_not_concluded_tests(tests)
    solutions_responses = {participant: {} for participant in tests.keys()}
    for id in tests:
        for test_idx, test in enumerate(tests[id]):
            solutions_responses[id][test_idx] = pd.DataFrame()
            solutions = []
            responses = []
            for dict in tests[id][test_idx]:
                if 'solution' in dict:
                    solutions.append(dict['solution'])
                elif 'response' in dict:
                    responses.append(dict['response'])
            solutions_responses[id][test_idx]['solutions'] = solutions
            solutions_responses[id][test_idx]['responses'] = responses
            solutions_responses[id][test_idx].dropna(axis=0, inplace=True)
    return solutions_responses


def delete_not_concluded_tests(tests):
    for id in tests:
        for test_idx, test in enumerate(tests[id]):
            if 'sequence' not in tests[id][test_idx][-1]:
                del tests[id][test_idx]
    return tests


def get_mse(root_path, all_tests):
    solutions_responses = get_solutions_and_responses(root_path=root_path)
    mse = {participant: {} for participant in solutions_responses.keys()}
    if not all_tests:
        for id in solutions_responses:
            for test in solutions_responses[id]:
                mse[id][str(test)] = mean_squared_error(solutions_responses[id][test]['responses'], solutions_responses[id][test]['solutions'])
        return mse
    if all_tests:
        tests_combined = {participant: {} for participant in solutions_responses.keys()}
        for id in solutions_responses:
            tests_combined[id] = pd.DataFrame()
            for tests in solutions_responses[id]:
                tests_combined[id] = tests_combined[id]._append(solutions_responses[id][tests])
                # tests_combined[id] = pd.concat([tests_combined[id]], pd.DataFrame(solutions_responses[id][tests]), ignore_index=True)
        for id in tests_combined:
            mse[id] = mean_squared_error(tests_combined[id]['responses'], tests_combined[id]['solutions'])
        return mse


def get_lin_reg(root_path, all_tests):
    solutions_responses = get_solutions_and_responses(root_path=root_path)
    lin_reg = {participant: {} for participant in solutions_responses.keys()}
    if not all_tests:
        for id in solutions_responses:
            for test in solutions_responses[id]:
                lin_reg[id][str(test)] = linregress(x=solutions_responses[id][test])
        return lin_reg
    if all_tests:
        tests_combined = {participant: {} for participant in solutions_responses.keys()}
        for id in solutions_responses:
            tests_combined[id] = pd.DataFrame()
            for tests in solutions_responses[id]:
                tests_combined[id] = tests_combined[id]._append(solutions_responses[id][tests])
        for id in tests_combined:
            lin_reg[id] = linregress(x=tests_combined[id])
        return lin_reg


def convert_ve_format(root_path):
    file_paths = get_file_paths(root_path=root_path)
    for participant, file_path_list in file_paths.items():
        for file_path in file_path_list:
            stage_tag = slab.ResultsFile.read_file(file_path, "stage")
            if not stage_tag:
                stem = file_path.stem
                stage = stem[len(participant) + 1: stem.find("_2022")]
                stem_without_stage = stem[:len(participant)] + stem[stem.find("_2022"):]
                out_file_path = file_path.parent / (stem_without_stage + file_path.suffix)
                open(out_file_path, "w").write(json.dumps({"stage": stage}) + '\n' + open(file_path).read())
                print(out_file_path)
            else:
                print(file_path.stem, "is already converted")


def move_old_ve_results_files(root_path):
    file_paths = get_file_paths(root_path=root_path)
    for participant, file_path_list in file_paths.items():
        for file_path in file_path_list:
            stage_tag = slab.ResultsFile.read_file(file_path, "stage")
            if not stage_tag:
                old_folder_path = file_path.parent.parent / "results_files_old"
                if not os.path.exists(old_folder_path):
                    os.makedirs(old_folder_path)
                os.rename(file_path, old_folder_path / file_path.name)

''''
sns.regplot(data=solutions_responses['03d3rc'][0], x='solutions', y='responses')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.show()
solutions_responses[]
'''







