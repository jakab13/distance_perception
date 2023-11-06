import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from analysis.distance_plasticity.utils import load_df, phase_to_text

folder_path = pathlib.Path(os.getcwd()) / "analysis" / "distance_plasticity"

df_distance_discrimination, df_visual_mapping = load_df()

subjects = df_distance_discrimination.subject_ID.unique()
spk_dists = sorted(df_distance_discrimination.spk_dist.unique())

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
for spk_idx, spk_dist in enumerate(sorted(df_distance_discrimination["spk_dist"].unique())):
    df = df_distance_discrimination[df_distance_discrimination["spk_dist"] == spk_dist]
    sns.kdeplot(data=df, x="signed_err", hue="block", ax=ax[spk_idx], common_norm=False, fill=True, palette="crest", alpha=.5, linewidth=0)
    ax[spk_idx].get_legend().remove()
    ax[spk_idx].set_title(str(spk_dist))

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

for subject_idx, subject in enumerate(sorted(subjects[-2:])):
    df_sub = df_distance_discrimination[df_distance_discrimination.subject_ID == subject]
    # df_sub = df_distance_discrimination
    # subject = "all subjects"
    fig, ax = plt.subplots(nrows=1, ncols=len(df_sub.phase.unique()), sharex=True, sharey=True, figsize=(20, 4))
    for phase in range(len(df_distance_discrimination.phase.unique())):
        df = df_sub[df_distance_discrimination.phase == phase]
        min = df_distance_discrimination.spk_dist.min()
        max = df_distance_discrimination.spk_dist.max()
        linreg = scipy.stats.linregress(df.spk_dist.dropna(), df.slider_dist.dropna())
        sns.regplot(data=df, x="spk_dist", y="slider_dist", ax=ax[phase], scatter_kws={"alpha": 0.15})
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
    # plt.savefig(folder_path / "figures" / title, dpi=400, overwrite=True)
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
        match block:
            case 0:
                axis_title = "Pre"
            case 1:
                axis_title = "Post-1"
            case 2:
                axis_title = "Post-2"
            case 3:
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
