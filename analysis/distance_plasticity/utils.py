import pandas as pd
import slab
import pathlib
import os
from datetime import datetime

pd.options.mode.chained_assignment = None
_phase_to_text_converter = {
    0: "pre",
    1: "post-1",
    2: "post-2",
    3: "post-3"
}


def load_df():
    results_folder = pathlib.Path(os.getcwd()) / "analysis" / "distance_plasticity" / "results"

    subjects_excl = ["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "varvara"]

    subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
    subjects = sorted([s for s in subjects if not any(s in excl for excl in subjects_excl)])

    results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

    visual_mapping_columns = ["subject_ID", "visual_obj_dist", "slider_val", "slider_ratio", "slider_dist", "response_time"]
    distance_discrimination_columns = ["subject_ID", "block", "spk_dist", "channel", "slider_val", "slider_ratio", "slider_dist", "response_time", "USO_file_name"]
    reverb_pse_columns = ["subject_ID", "block", "start_reverse_t60", "end_reverse_t60", "n_loop", "n_ref_played"]

    df_visual_mapping = pd.DataFrame(columns=visual_mapping_columns)
    df_distance_discrimination = pd.DataFrame(columns=distance_discrimination_columns)
    dr_reverb_pse = pd.DataFrame(columns=reverb_pse_columns)

    block_to_phase = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}

    for subject, results_file_list in results_files.items():
        distance_discrimination_block_counter = 0
        reverb_pse_block_counter = 0
        for results_file_name in results_file_list:
            path = results_folder / subject / results_file_name
            date_time = _get_date_time(path)
            stage = slab.ResultsFile.read_file(path, tag="stage")
            df_curr = pd.DataFrame()
            if stage == "visual_mapping":
                df_curr["subject_ID"] = [subject for _ in range(len(slab.ResultsFile.read_file(path, tag="visual_obj_dist")))]
                df_curr["visual_obj_dist"] = slab.ResultsFile.read_file(path, tag="visual_obj_dist")
                df_curr["slider_val"] = slab.ResultsFile.read_file(path, tag="slider_val")
                df_curr["slider_ratio"] = slab.ResultsFile.read_file(path, tag="slider_ratio")
                df_curr["slider_dist"] = slab.ResultsFile.read_file(path, tag="slider_dist")
                df_curr["response_time"] = slab.ResultsFile.read_file(path, tag="response_time")
                df_visual_mapping = pd.concat([df_visual_mapping, df_curr], ignore_index=True)
            if stage == "distance_discrimination_task":
                block_length = slab.ResultsFile.read_file(path, tag="seq")["n_trials"]
                df_curr["idx"] = range(block_length)
                df_curr["subject_ID"] = [subject for _ in range(block_length)]
                df_curr["block"] = [distance_discrimination_block_counter for _ in range(block_length)]
                df_curr["phase"] = block_to_phase[distance_discrimination_block_counter]
                df_curr["spk_dist"] = slab.ResultsFile.read_file(path, tag="spk_dist")
                df_curr["spk_dist_delta"] = df_curr["spk_dist"].diff()
                df_curr["channel"] = slab.ResultsFile.read_file(path, tag="channel")
                df_curr["slider_val"] = slab.ResultsFile.read_file(path, tag="slider_val")
                df_curr["slider_ratio"] = slab.ResultsFile.read_file(path, tag="slider_ratio")
                df_curr["slider_dist"] = slab.ResultsFile.read_file(path, tag="slider_dist")
                df_curr["slider_dist_delta"] = df_curr["slider_dist"].diff()
                df_curr["response_time"] = slab.ResultsFile.read_file(path, tag="response_time")
                df_curr["USO_file_name"] = slab.ResultsFile.read_file(path, tag="USO_file_name")
                df_curr["slider_min"] = slab.ResultsFile.read_file(path, tag="slider_min_max")[0]
                df_curr["slider_max"] = slab.ResultsFile.read_file(path, tag="slider_min_max")[1]
                # df_curr["date_time"] = date_time
                # df_curr["timestamp"] = pd.to_datetime(df_curr["date_time"].astype(int)) + df_curr["response_time"]
                df_distance_discrimination = pd.concat([df_distance_discrimination, df_curr], ignore_index=True)
                distance_discrimination_block_counter += 1
            if stage == "reverb_pse":
                block_length = len(slab.ResultsFile.read_file(path, tag="start_reverse_t60"))
                df_curr["subject_ID"] = [subject for _ in range(block_length)]
                df_curr["block"] = [reverb_pse_block_counter for _ in range(block_length)]
                df_curr["start_reverse_t60"] = slab.ResultsFile.read_file(path, tag="start_reverse_t60")
                df_curr["end_reverse_t60"] = slab.ResultsFile.read_file(path, tag="end_reverse_t60")
                df_curr["n_loop"] = slab.ResultsFile.read_file(path, tag="n_loop")
                df_curr["n_ref_played"] = slab.ResultsFile.read_file(path, tag="n_ref_played")
                dr_reverb_pse = pd.concat([dr_reverb_pse, df_curr], ignore_index=True)
                reverb_pse_block_counter += 1

    # Dropping first 3 trials in naive test
    df_distance_discrimination = df_distance_discrimination.drop(
        df_distance_discrimination[
            (df_distance_discrimination["block"] == 0) & (df_distance_discrimination["idx"] < 3)].index)

    # for dist in df_distance_discrimination.spk_dist.unique():
    #     df_distance_discrimination["spk_dist"].replace(
    #         dist, round(np.interp(dist, [2.1, 11.78], [2.0, 12.0]), 2), inplace=True
    #     )

    df_distance_discrimination["spk_dist"].replace(2, 2.1, inplace=True)

    # Filtering for timing outliers
    df_distance_discrimination = df_distance_discrimination[df_distance_discrimination["response_time"].between(0.3, 10)]

    # Calculating and storing errors
    df_distance_discrimination["signed_err"] = (df_distance_discrimination["slider_dist"] - df_distance_discrimination["spk_dist"]).astype('float64')
    df_distance_discrimination["absolute_err"] = df_distance_discrimination["signed_err"].abs()
    df_distance_discrimination["signed_err_2"] = df_distance_discrimination["signed_err"]**2

    # Calculating and storing errors (visual mapping)
    df_visual_mapping["signed_err"] = (df_visual_mapping["slider_dist"] - df_visual_mapping["visual_obj_dist"]).astype('float64')
    df_visual_mapping["absolute_err"] = df_visual_mapping["signed_err"].abs()
    df_visual_mapping["signed_err_2"] = df_visual_mapping["signed_err"]**2

    return df_distance_discrimination, df_visual_mapping


def phase_to_text(phase):
    return _phase_to_text_converter[phase]


def _get_date_time(path):
    date_time_string = path.stem[7:]
    date_time = datetime.strptime(date_time_string, '%Y-%m-%d-%H-%M-%S')
    return date_time