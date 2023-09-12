import pandas as pd
import slab
import pathlib
import os
import seaborn as sns


results_folder = pathlib.Path(os.getcwd()) / "analysis" / "distance_plasticity" / "results"

subjects = [f for f in os.listdir(results_folder) if not f.startswith('.')]

results_files = {s: [f for f in sorted(os.listdir(results_folder / s))] for s in subjects}

df_visual_mapping = pd.DataFrame()
df_distance_discrimination = pd.DataFrame()

for subject, results_file_list in results_files.items():
    for results_file_name in results_file_list:
        path = results_folder / subject / results_file_name
        stage = slab.ResultsFile.read_file(path, tag="stage")
        if stage == "visual_mapping":
            df_visual_mapping["subject_ID"] = [subject for _ in range(len(slab.ResultsFile.read_file(path, tag="visual_obj_dist")))]
            df_visual_mapping["visual_obj_dist"] = slab.ResultsFile.read_file(path, tag="visual_obj_dist")
            df_visual_mapping["slider_val"] = slab.ResultsFile.read_file(path, tag="slider_val")
            df_visual_mapping["slider_ratio"] = slab.ResultsFile.read_file(path, tag="slider_ratio")
            df_visual_mapping["slider_dist"] = slab.ResultsFile.read_file(path, tag="slider_dist")
            df_visual_mapping["response_time"] = slab.ResultsFile.read_file(path, tag="response_time")

print(df_visual_mapping)

visual_mapping_plot = sns.regplot(x=df_visual_mapping["visual_obj_dist"], y=df_visual_mapping["slider_dist"])
visual_mapping_plot.set(title="Visual mapping")