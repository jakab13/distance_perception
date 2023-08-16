from result_file_functions import get_all_results, \
    get_number_of_tests, get_number_of_trainings, get_score, get_mse, get_lin_reg, get_rea_times
import pandas as pd
from statistics import mean, median
root_path = '/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/data/vocal_effort'
result_files = get_all_results(root_path=root_path)
only_concluded_tests = True


def create_behavioural_dataframe(root_path=root_path, only_concluded_tests=True):
    result_files = get_all_results(root_path)
    df = pd.DataFrame()
    df['participant_id'] = result_files.keys()

    n_trainings = get_number_of_trainings(result_files=result_files)
    for id in result_files.keys():
        df.loc[df['participant_id'] == id, 'n_trainings'] = n_trainings[id]

    n_tests = get_number_of_tests(result_files=result_files)
    for id in result_files.keys():
        df.loc[df['participant_id'] == id, 'n_tests'] = n_tests[id]

    scores = get_score(only_concluded_tests=only_concluded_tests, root_path=root_path, result_files=result_files)
    for id in scores:
        for test in scores[id]:
            df.loc[df['participant_id'] == id, 'score_' + test + '[%]'] = scores[id][test]

    mse = get_mse(root_path=root_path, all_tests=False)
    for id in mse:
        for test in mse[id]:
            df.loc[df['participant_id'] == id, 'mse_test_' + test] = mse[id][test]

    lin_reg = get_lin_reg(root_path=root_path, all_tests=False)
    for id in lin_reg:
        for test in lin_reg[id]:
            df.loc[df['participant_id'] == id, 'reg_slope_' + test] = lin_reg[id][test].slope
            df.loc[df['participant_id'] == id, 'reg_intercept_' + test] = lin_reg[id][test].intercept
            df.loc[df['participant_id'] == id, 'reg_rvalue_' + test] = lin_reg[id][test].rvalue
            df.loc[df['participant_id'] == id, 'reg_pvalue_' + test] = lin_reg[id][test].pvalue
            df.loc[df['participant_id'] == id, 'reg_stderr_' + test] = lin_reg[id][test].stderr
            df.loc[df['participant_id'] == id, 'reg_intercept_stderr_' + test] = lin_reg[id][test].intercept_stderr

    for id in scores:
        all_tests = []
        for test in scores[id]:
            all_tests.append(scores[id][test])
        df.loc[df['participant_id'] == id, 'best_score'] = max(all_tests)
        df.loc[df['participant_id'] == id, 'worst_score'] = min(all_tests)
        df.loc[df['participant_id'] == id, 'avg_score'] = mean(all_tests)
        df.loc[df['participant_id'] == id, 'median_score'] = median(all_tests)

    mse_tests_combined = get_mse(root_path=root_path, all_tests=True)
    for id in mse_tests_combined:
        df.loc[df['participant_id'] == id, 'mse_tests_combined'] = mse_tests_combined[id]

    lin_reg_combined = get_lin_reg(root_path=root_path, all_tests=True)
    for id in lin_reg_combined:
        df.loc[df['participant_id'] == id, 'reg_slope_tests_combined'] = lin_reg_combined[id].slope

    rea_times = get_rea_times(root_path=root_path, only_concluded_tests=only_concluded_tests)
    for id in rea_times:
        mean_rea_times = []
        for test in rea_times[id]:
            df.loc[df['participant_id'] == id, 'avg_react_time_' + test + '[ms]'] = mean(rea_times[id][test])
            mean_rea_times.append(mean(rea_times[id][test]))
        df.loc[df['participant_id'] == id, 'avg_react_time_tests_combined[ms]'] = mean(mean_rea_times)
    return df


















