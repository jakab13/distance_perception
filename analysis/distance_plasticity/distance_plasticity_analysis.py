import copy
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy
import random
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from analysis.distance_plasticity.utils import load_df, phase_to_text, bs, bs_diff, ci_data, expected_value

folder_path = pathlib.Path(os.getcwd()) / "analysis" / "distance_plasticity"

df_distance_discrimination, df_visual_mapping = load_df()

subjects = df_distance_discrimination.subject_ID.unique()
spk_dists = sorted(df_distance_discrimination.spk_dist.unique())

palette_tab10 = sns.color_palette(palette="tab10")

visual_mapping_plot = sns.regplot(data=df_visual_mapping, x="visual_obj_dist", y="slider_dist")
visual_mapping_plot.set(title="Visual mapping")

# Distribution of distance estimation as a function of speaker distance
fig, ax = plt.subplots(nrows=len(spk_dists), ncols=1, sharex=True)
for spk_idx, spk_dist in enumerate(spk_dists):
    df = df_distance_discrimination[df_distance_discrimination.spk_dist == spk_dist]
    df.hist(column="signed_err", bins=50, xrot=90, ax=ax[spk_idx])
    ax[spk_idx].set_title(str(spk_dist) + "m")
    ax[spk_idx].grid = False
fig.suptitle("Signed error distribution at different distances")

# Distribution of response times as a function of subject
df_distance_discrimination.hist(column="response_time", by="subject_ID", bins=50, sharex=True, xrot=90, layout=(len(subjects), 1))

# Distribution of response times at different blocks as a function of subject
fig, ax = plt.subplots(nrows=len(subjects), ncols=1, sharex=True)
for subject_idx, subject in enumerate(sorted(subjects)):
    df = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    sns.kdeplot(data=df, x="response_time", hue="block", ax=ax[subject_idx], common_norm=False, fill=True, palette="crest", alpha=.5, linewidth=0)
    ax[subject_idx].get_legend().remove()
    ax[subject_idx].set_title(subject)

# Distribution of signed error at different distances and blocks
fig, ax = plt.subplots(nrows=11, ncols=1, sharex=True)
for spk_idx, spk_dist in enumerate(spk_dists):
    df = df_distance_discrimination[df_distance_discrimination["spk_dist"] == spk_dist]
    sns.kdeplot(data=df, x="signed_err", hue="phase", ax=ax[spk_idx], common_norm=False, fill=True, palette="tab10")
    ax[spk_idx].get_legend().remove()
    ax[spk_idx].set_title(str(spk_dist))

sns.boxplot(df_distance_discrimination, x="spk_dist", y="signed_err", hue="phase", width=0.5)\
    .set(title="Aggregated error at different phases",
         xlabel="Presented distance (m)",
         ylabel="Error (m)")

sns.displot(data=df_distance_discrimination, x="signed_err", hue="phase", kind="kde", row="spk_dist", palette="tab10",
            height=1, common_norm=False, aspect=5)

avg_error_per_subject = df_distance_discrimination.groupby(["spk_dist", 'subject_ID', 'phase'])["signed_err"].agg(['mean', 'var', 'std']).reset_index()

sns.catplot(data=avg_error_per_subject, x="spk_dist", y="std", hue="phase", kind="bar", col="subject_ID", col_wrap=4)

# Absolute error at different blocks
sns.pointplot(data=df_distance_discrimination, x="block", y="absolute_err", errorbar="se")
plt.xlabel("Phase")
plt.ylabel("Absolute Error (m)")
plt.xticks([0, 1, 2, 3], ["Pre", "Post-1", "Post-2", "Post-3"])
plt.title("Mean absolute error before/after training sessions")

# Absolute error at different blocks per subject
sns.pointplot(data=df_distance_discrimination, x="block", y="absolute_err", hue="subject_ID", errorbar="se")
plt.xlabel("Phase")
plt.ylabel("Absolute Error (m)")
plt.xticks([0, 1, 2, 3], ["Pre", "Post-1", "Post-2", "Post-3"])
plt.title("Mean absolute error before/after training sessions")

for subject_idx, subject in enumerate(sorted(subjects[14:15])):
    df_sub = df_distance_discrimination[df_distance_discrimination.subject_ID == subject]
    # df_sub = df_distance_discrimination
    # subject = f"N={len(df_distance_discrimination.subject_ID.unique())}"
    fig, ax = plt.subplots(nrows=1, ncols=len(df_sub.phase.unique()), sharex=True, sharey=True, figsize=(20, 4))
    for phase in range(len(df_distance_discrimination.phase.unique())):
        df = df_sub[df_distance_discrimination.phase == phase]
        min = df_distance_discrimination.spk_dist.min()
        max = df_distance_discrimination.spk_dist.max()
        linreg = scipy.stats.linregress(df.spk_dist.dropna(), df.slider_dist.dropna())
        sns.regplot(data=df, x="spk_dist", y="slider_dist", ax=ax[phase], scatter=False, scatter_kws={"alpha": 0.01}, ci=95)
        # sns.regplot(data=df.groupby(by=["subject_ID", "spk_dist"], as_index=False)["slider_dist"].mean(),
        #             x="spk_dist", y="slider_dist", ax=ax[phase], scatter=False, scatter_kws={"alpha": 0.01})
        # sns.lineplot(data=df, x="spk_dist", y="slider_dist", ax=ax[phase], errorbar=("ci", 95))
        ax[phase].plot(
            np.arange(min, max + 1),
            np.arange(min, max + 1),
            color="grey", linestyle="--")
        axis_title = phase_to_text(phase)
        ax[phase].set_xlabel("Presented (m)")
        ax[phase].set_ylabel("Perceived (m)")
        ax[phase].set_xlim([1, 13])
        ax[phase].set_ylim([1, 13])
        ax[phase].set_title(phase_to_text(phase))
        textstr = '\n'.join((
            f'slope={linreg.slope:.2f}',
            f'R2={linreg.rvalue**2:.2f}'))
        ax[phase].text(0.05, 0.95, textstr, transform=ax[phase].transAxes, verticalalignment='top')
    title = "Presented vs Perceived throughout training (" + subject + ")"
    # plt.tight_layout()
    fig.suptitle(title)
    plt.savefig(folder_path / "figures" / title, dpi=400)
    plt.savefig(folder_path / "figures" / str(title + ".eps"), format="eps")
    # plt.close()
    plt.show()

sns.scatterplot(data=df_distance_discrimination, x="subject_ID", y="slider_max")


for subject_idx, subject in enumerate(sorted(subjects)):
    df_sub = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    fig, ax = plt.subplots(nrows=1, ncols=len(df_sub["block"].unique()), sharex=True, sharey=True, figsize=(16, 6))
    for block in range(len(df_distance_discrimination["block"].unique())):
        ax[block].hlines(0, xmin=0, xmax=12, colors="grey", linestyles="--", alpha=0.5)
        df = df_sub[df_sub["block"] == block]
        mse = mean_squared_error(df['slider_dist'], df['spk_dist'])
        mae = mean_absolute_error(df['slider_dist'], df['spk_dist'])
        signed_err_mean = df["signed_err"].mean()
        signed_err_std = df["signed_err"].std()

        ax[block].hlines(signed_err_mean, xmin=0, xmax=11, colors="red", alpha=0.5)
        std_line_top = (signed_err_mean + signed_err_std) * np.ones(11)
        std_line_bottom = (signed_err_mean - signed_err_std) * np.ones(11)
        ax[block].fill_between(np.arange(11), std_line_top, std_line_bottom, color="orange", alpha=0.1)
        sns.pointplot(data=df, x="spk_dist", y="signed_err", ax=ax[block], errorbar="se")
        if block == 0:
            axis_title = "Pre"
        if block == 0:
            axis_title = "Post-1"
        if block == 0:
            axis_title = "Post-2"
        if block == 0:
            axis_title = "Post-3"
        ax[block].set_xticks([0, 5, 10])
        ax[block].set_xticklabels([2, 7, 12])
        ax[block].set_title(axis_title + " " + "MSE=" + str(round(mse, 2)))
        ax[block].set_xlabel("Speaker distance (m)")
        ax[block].set_ylabel("Signed error (m)")
        ax[block].set_ylim([-5, 5])
    title = "Signed error vs Distance in different training phases (" + subject + ")"
    fig.suptitle(title)
    plt.savefig(folder_path / "figures" / title)
    # plt.close(fig)
    plt.show()

df_visual_mapping.hist(column="signed_err", by="visual_obj_dist", bins=20, sharex=True, xrot=90, layout=(10, 1))

distance_discrimination_plot = sns.regplot(data=df_distance_discrimination, x="spk_dist", y="slider_dist")

ax = sns.boxplot(x='block', y='signed_err', data=df_distance_discrimination, hue="subject_ID")
ax = sns.stripplot(x='block', y='signed_err', data=df_distance_discrimination, hue="subject_ID")

fig, ax = plt.subplots()
sns.boxplot(x='block', y='signed_err', data=df_distance_discrimination, hue="subject_ID", ax=ax)
for subject in subjects:
    df_subject = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    sns.regplot(x='block', y='signed_err', data=df_subject, ax=ax, scatter=False)

sns.lmplot(x='block', y='signed_err', data=df_distance_discrimination, hue="subject_ID", scatter=False)

# Ordinary Least Squares (OLS) model
for subject in subjects:
    df_subject = df_distance_discrimination[df_distance_discrimination["subject_ID"] == subject]
    model = smf.ols('signed_err ~ C(block)', data=df_subject).fit()
    mse = model.mse_resid
    # print("MSE", mse)
    anova_table = sm.stats.anova_lm(model, typ=2)
    # linreg = scipy.stats.linregress(df_subject["block"], df_subject["signed_err"])
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

md = smf.mixedlm("signed_err ~ C(block)", df_distance_discrimination, groups="subject_ID")
mdf = md.fit()
print(mdf.summary())

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# Load the CSV file
data = pd.read_csv('/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/distance_plasticity.csv')

# Exclude subjects 31 and 32
data = data[~data['subject_ID'].isin(['sub_31', 'sub_32'])]

# List to store individual betas
individual_betas_list = []

# Iterate over each subject and phase to calculate individual betas
for subject in data['subject_ID'].unique():
    for phase in data['phase'].unique():
        subject_phase_data = data[(data['subject_ID'] == subject) & (data['phase'] == phase)]
        if not subject_phase_data.empty:
            slope, intercept, r_value, p_value, std_err = stats.linregress(subject_phase_data['spk_dist'], subject_phase_data['slider_dist'])
            individual_betas_list.append({'subject_ID': subject, 'phase': phase, 'slope': slope})

# Convert list to DataFrame
individual_betas = pd.DataFrame(individual_betas_list)

# Convert 'phase' to an integer or categorical type
individual_betas['phase'] = individual_betas['phase'].astype(int)
# Alternatively, convert to a categorical type
# individual_betas['phase'] = individual_betas['phase'].astype('category')

# Convert 'phase' and 'subject_ID' to categorical types
individual_betas['phase'] = individual_betas['phase'].astype('category')
individual_betas['subject_ID'] = individual_betas['subject_ID'].astype('category')

# Fit the model for two-way ANOVA
model = ols('slope ~ C(subject_ID) + C(phase)', data=individual_betas).fit()


anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

fig, ax = plt.subplots(nrows=len(spk_dists), ncols=1, sharex=True, figsize=(15, 25))
for spk_idx, spk_dist in enumerate(spk_dists):
    df = df_distance_discrimination[df_distance_discrimination.spk_dist == spk_dist]
    ax_curr = ax[spk_idx]
    ax_curr.set_title(f"Presented distance: {spk_dist}m")
    ax_curr.set_xlabel("Signed error (m)")
    ax_curr.set_xlim(-6, 6)
    sns.histplot(df,
                x="signed_err",
                hue="phase",
                common_norm=False,
                palette="tab10",
                bins=20,
                multiple="dodge",
                shrink=.6,
                stat="density",
                ax=ax_curr,
                 alpha=.2,
                 lw=0
                )
    sns.kdeplot(df,
                x="signed_err",
                hue="phase",
                palette="tab10",
                common_norm=False,
                ax=ax_curr,
                alpha=0
                )
    y_max = np.max([l.get_ydata() for l in ax_curr.lines])
    patches = []
    for phasex_idx, phase in enumerate(df.phase.unique()):
        df_phase = df[df.phase == phase]
        mean = df_phase["signed_err"].mean()
        median = np.median(df_phase["signed_err"])
        stdev = df_phase["signed_err"].std()
        color = palette_tab10[phasex_idx]
        left = mean - stdev
        right = mean + stdev
        x_pdf = np.linspace(df_phase["signed_err"].min(), df_phase["signed_err"].max(), 100)
        y_pdf = scipy.stats.norm.pdf(x_pdf, mean, stdev)
        ax_curr.vlines(mean, 0, y_pdf.max(), color=color)
        ax_curr.vlines(median, 0, y_pdf.max(), color=color, ls=":")
        ax_curr.plot(x_pdf, y_pdf, color=color)
        line_patch = mpatches.Patch(color=color, label=f"phase {int(phase)}")
        patches.append(line_patch)
        # ax_curr.fill_between(x_pdf, 0, y_pdf, where=(left < x_pdf) & (x_pdf <= right), interpolate=True, facecolor=color, alpha=0.1)
    ax_curr.legend(handles=patches)
main_title = "Signed error distributions at different phases"
fig.tight_layout()
plt.savefig(main_title, dpi=400, overwrite=True)
plt.show()


for spk_idx, spk_dist in enumerate(spk_dists):
    df = df_distance_discrimination[
        (df_distance_discrimination.spk_dist == spk_dist)
        # (df_distance_discrimination.phase != 0.0)
    ]
    g = sns.FacetGrid(df, row="phase", hue="phase", palette=palette_tab10, sharex=True, sharey=True, height=2, aspect=5, xlim=(-6, 6))
    g.map(sns.histplot, "signed_err", bins=50, kde=True)
    title = f"Signed error at different phases (presented distance at {spk_dist}m)"
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(title + ".png", format="png", dpi=400, overwrite=True)
    plt.show()


fig, ax = plt.subplots(nrows=len(spk_dists), ncols=1, sharex=True, figsize=(15, 25))
for spk_idx, spk_dist in enumerate(spk_dists):
    ax_curr = ax[spk_idx]
    for phase_idx in [1, 2, 3]:
        phase_data = df_distance_discrimination[
            (df_distance_discrimination.spk_dist == spk_dist) &
            (df_distance_discrimination.phase == phase_idx)
        ]["signed_err"]
        bs_array = bs(phase_data)
        bs_mean, bs_median, bs_se, bs_lower_ci, bs_upper_ci = ci_data(bs_array)
        distribution_line = sns.kdeplot(bs_array, color=palette_tab10[phase_idx], ax=ax_curr)
        ax_curr.axvline(bs_mean, color=palette_tab10[phase_idx], label=f"Phase {phase_idx}")
        # ax_curr.axvline(bs_lower_ci, ymax= linestyle="--", color=palette_tab10[phase_idx])
        # ax_curr.axvline(bs_upper_ci, ymax= linestyle="--", color=palette_tab10[phase_idx])
    ax_curr.set_title(f"{spk_dist}m")
    ax_curr.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
title = "Bootstrap distribution of median signed error values in the different phases"
plt.suptitle(title, y=1)
plt.tight_layout()
# plt.savefig(title + ".png", format="png", dpi=400, overwrite=True)
plt.show()


df_bs = pd.DataFrame(columns=["subject_ID", "phase", "spk_dist", "slider_dist_bs_mean", "slider_dist_median", "slider_dist_expected_val"])
for subject in subjects:
    df_sub = df_distance_discrimination[df_distance_discrimination.subject_ID == subject]
    print(f"Starting with subject: {subject}")
    for spk_dist in spk_dists:
        df_dist = df_sub[df_sub.spk_dist == spk_dist]
        # df_phase_0 = df_dist[df_dist.phase == 0]["slider_dist"]
        df_phase_1 = df_dist[df_dist.phase == 1.0]["slider_dist"]
        for phase in [1.0, 2.0, 3.0]:
            df_phase_curr = df_dist[df_dist.phase == phase]["slider_dist"]
            # bs_slider_dist_dist = bs_diff(df_phase_curr, df_phase_1)
            # bs_slider_dist_mean = bs_slider_dist_dist.mean()
            bs_slider_dist_mean = None
            slider_median = df_phase_curr.median() - df_phase_1.median()
            slider_expected_val = expected_value(df_phase_curr) - expected_value(df_phase_1)
            df_bs.loc[-1] = [subject, phase, spk_dist, bs_slider_dist_mean, slider_median, slider_expected_val]
            df_bs.index += 1
        print(f"Done with distance: {spk_dist}")
df_bs = df_bs.sort_index()

df_linreg = pd.DataFrame(columns=["subject_ID", "phase", "beta_bs", "beta_median"])
for subject in subjects:
    df_bs_sub = df_bs[df_bs.subject_ID == subject]
    for phase in [1, 2, 3]:
        df_bs_curr_phase = df_bs_sub[df_bs_sub.phase == phase]
        linreg_bs = scipy.stats.linregress(df_bs_curr_phase["spk_dist"], df_bs_curr_phase["slider_dist_bs_mean"])
        linreg_median = scipy.stats.linregress(df_bs_curr_phase["spk_dist"], df_bs_curr_phase["slider_dist_median"])
        beta_bs = linreg_bs.slope
        beta_median = linreg_median.slope
        df_linreg.loc[-1] = [subject, phase, beta_bs, beta_median]
        df_linreg.index += 1
df_linreg = df_linreg.sort_index()

ax = sns.lmplot(df_bs, x="spk_dist", y="slider_dist_median", hue="phase", scatter=False)
ax.set(xlabel="Presented (m)", ylabel="Normalised Perceived (m)")
plt.title("Linear regressions of median perceived \n distances normalised to phase 1")
plt.tight_layout()
plt.savefig("Slopes normalised to phase 1", dpi=400)

linreg_bs_distance_compression = scipy.stats.linregress(df_linreg["phase"].astype(float), df_linreg["beta_bs"])
linreg_median_distance_compression = scipy.stats.linregress(df_linreg["phase"].astype(float), df_linreg["beta_median"])

ax = sns.barplot(df_linreg, x="phase", y="beta_median")
ax.set_title("Slope values normalised to phase 1")
ax.set(xlabel="Phase", ylabel="Normalised Slope")
ax.text(0.05, 0.95, '\n'.join(("Linreg p-value:", f"{round(linreg_median_distance_compression.pvalue, 5)}")), transform=ax.transAxes, verticalalignment='top')
plt.savefig(folder_path / "figures" / "Slope values normalised to phase 1", dpi=400)

# Long session analysis
df_long_sess = df_distance_discrimination[df_distance_discrimination.subject_ID.isin(["sub_long_sess_1", "sub_long_sess_2", "sub_long_sess_3", "sub_long_sess_4", "sub_long_sess_5"])]
sns.boxplot(df_long_sess, x="subject_ID", y="absolute_err", hue="phase")

df_long_sess_performance = pd.DataFrame(columns=["subject_ID", "phase", "beta", "stderr"])
for subject_ID in df_long_sess.subject_ID.unique():
    for phase in df_long_sess.phase.unique()[1:]:
        df_long_sess_phase = df_long_sess[(df_long_sess.phase == phase) & (df_long_sess.subject_ID == subject_ID)]
        linreg = scipy.stats.linregress(df_long_sess_phase["spk_dist"].astype(float), df_long_sess_phase["slider_dist"].astype(float))
        beta = linreg.slope
        stderr = linreg.stderr
        df_long_sess_performance.loc[-1] = [subject_ID, phase, beta, stderr]
        df_long_sess_performance.index += 1
# df_long_sess_performance = df_long_sess_performance.sort_index()
sns.barplot(df_long_sess_performance, x="subject_ID", y="beta")
