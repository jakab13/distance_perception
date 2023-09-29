import pandas as pd
import slab
import pathlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error

results_folder = pathlib.Path(os.getcwd()) / "analysis" / "distance_plasticity" / "results"

subjects_excl = ["sub_01", "sub_02", "sub_03", "sub_04", "varvara"]

subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
subjects = [s for s in subjects if not any(s in excl for excl in subjects_excl)]

results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

visual_mapping_columns = ["subject_ID", "visual_obj_dist", "slider_val", "slider_ratio", "slider_dist", "response_time"]
distance_discrimination_columns = ["subject_ID", "block", "spk_dist", "channel", "slider_val", "slider_ratio", "slider_dist", "response_time", "USO_file_name"]
reverb_pse_columns = ["subject_ID", "block", "start_reverse_t60", "end_reverse_t60", "n_loop", "n_ref_played"]

df_visual_mapping = pd.DataFrame(columns=visual_mapping_columns)
df_distance_discrimination = pd.DataFrame(columns=distance_discrimination_columns)
dr_reverb_pse = pd.DataFrame(columns=reverb_pse_columns)

for subject, results_file_list in results_files.items():
    distance_discrimination_block_counter = 0
    reverb_pse_block_counter = 0
    for results_file_name in results_file_list:
        print(results_file_name)
        path = results_folder / subject / results_file_name
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
            df_curr["spk_dist"] = slab.ResultsFile.read_file(path, tag="spk_dist")
            df_curr["channel"] = slab.ResultsFile.read_file(path, tag="channel")
            df_curr["slider_val"] = slab.ResultsFile.read_file(path, tag="slider_val")
            df_curr["slider_ratio"] = slab.ResultsFile.read_file(path, tag="slider_ratio")
            df_curr["slider_dist"] = slab.ResultsFile.read_file(path, tag="slider_dist")
            df_curr["response_time"] = slab.ResultsFile.read_file(path, tag="response_time")
            df_curr["USO_file_name"] = slab.ResultsFile.read_file(path, tag="USO_file_name")
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

df_distance_discrimination["block"].replace(2, 1, inplace=True)
df_distance_discrimination["block"].replace(3, 2, inplace=True)
df_distance_discrimination["block"].replace(4, 2, inplace=True)
df_distance_discrimination["block"].replace(5, 3, inplace=True)
df_distance_discrimination["block"].replace(6, 3, inplace=True)

df_distance_discrimination = df_distance_discrimination[df_distance_discrimination["response_time"].between(0.3, 10)]

df_distance_discrimination["dist_diff"] = (df_distance_discrimination["slider_dist"] - df_distance_discrimination["spk_dist"]).astype('float64')
df_distance_discrimination["dist_diff_2"] = df_distance_discrimination["dist_diff"]**2

df_visual_mapping["dist_diff"] = (df_visual_mapping["slider_dist"] - df_visual_mapping["visual_obj_dist"]).astype('float64')
df_visual_mapping["dist_diff_2"] = df_visual_mapping["dist_diff"]**2

subjects = df_distance_discrimination["subject_ID"].unique()

print(df_distance_discrimination)

visual_mapping_plot = sns.regplot(data=df_visual_mapping, x="visual_obj_dist", y="slider_dist")
visual_mapping_plot.set(title="Visual mapping")

# Distribution of distance estimation as a function of subject
df_distance_discrimination.hist(column="dist_diff", by="spk_dist", bins=50, sharex=True, xrot=90, layout=(11, 1))

# Distribution of response times as a function subject
df_distance_discrimination.hist(column="response_time", by="subject_ID", bins=50, sharex=True, xrot=90, layout=(len(subjects), 1))

fig, ax = plt.subplots(nrows=11, ncols=1, sharex=True)
for spk_idx, spk_dist in enumerate(sorted(df_distance_discrimination["spk_dist"].unique())):
    df = df_distance_discrimination[df_distance_discrimination["spk_dist"] == spk_dist]
    sns.kdeplot(data=df, x="dist_diff", hue="block", ax=ax[spk_idx], common_norm=False, fill=True, palette="crest", alpha=.5, linewidth=0)
    ax[spk_idx].get_legend().remove()
    ax[spk_idx].set_title(str(spk_dist))

df_visual_mapping.hist(column="dist_diff", by="visual_obj_dist", bins=20, sharex=True, xrot=90, layout=(10, 1))

distance_discrimination_plot = sns.regplot(data=df_distance_discrimination, x="spk_dist", y="slider_dist")

# df_sub_1 = df_distance_discrimination[df_distance_discrimination["subject_ID"] == "sub_01"]
#
# ax = sns.boxplot(x='block', y='dist_diff', data=df_sub_1, color='#99c2a2')
# ax = sns.stripplot(x="block", y="dist_diff", data=df_sub_1, color='#7d0013')
# plt.show()

ax = sns.boxplot(x='block', y='dist_diff', data=df_distance_discrimination, hue="subject_ID")
ax = sns.stripplot(x='block', y='dist_diff', data=df_distance_discrimination, hue="subject_ID")

fig, ax = plt.subplots()
sns.boxplot(x='block', y='dist_diff', data=df_distance_discrimination, hue="subject_ID", ax=ax)
for subject in subjects:
    df_subject = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    sns.regplot(x='block', y='dist_diff', data=df_subject, ax=ax, scatter=False)

sns.lmplot(x='block', y='dist_diff', data=df_distance_discrimination, hue="subject_ID", scatter=False)

# Ordinary Least Squares (OLS) model
for subject in subjects:
    df_subject = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    model = smf.ols('dist_diff ~ C(block)', data=df_subject).fit()
    mse = model.mse_resid
    # print("MSE", mse)
    anova_table = sm.stats.anova_lm(model, typ=2)
    # linreg = scipy.stats.linregress(df_subject["block"], df_subject["dist_diff"])
    print("Subject:", subject)
    print("ANOVA", anova_table)
    # print("Linear regression", linreg.slope, linreg.rvalue, linreg.pvalue)

for subject in subjects:
    print(subject)
    df_subject = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    for block in range(4):
        df_block = df_subject[df_subject["block"] == block]
        mse = mean_squared_error(df_block['slider_dist'], df_block['spk_dist'])
        print(block, mse)

md = smf.mixedlm("dist_diff ~ C(block)", df_distance_discrimination, groups="subject_ID")
mdf = md.fit()
print(mdf.summary())
