import os
import plot_settings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pandas import NamedAgg

from data_utils import load_data_frame

idle_power = 6.1
power = {256: 7.4, 512: 7.5, 1024: 7.6}
dataset_num_examples = {"shd": 2264, "ssc": 20382}
num_sops = {("shd", 256): 7154159044, ("shd", 512): 18815875280,
            ("shd", 1024): 53417193820, ("ssc", 256): 65005252387, 
            ("ssc", 512): 168041369975, ("ssc", 1024): 488839537171}

# Load dataframe       
keys = ["dataset", "num_hidden"]
df = load_data_frame(keys, lambda p: p["dt"] == 1.0 and p["augmentation"] != "plain" and p["seed"] == 1234,
                     path=os.path.join("..", "classifier"),
                     load_test=True, load_test_perf=True)

# Sort
df = df.sort_values(["dataset", "num_hidden"])

# Calculate new columns
df["test_gpu_time"] = df["test_neuron_update_time"] + df["test_presynaptic_update_time"] + df["test_custom_update_reset_time"]
num_examples = df["dataset"].map(dataset_num_examples)
sim_power = df["num_hidden"].map(power)
num_sop = df.apply(lambda f: num_sops[(f["dataset"], f["num_hidden"])],
                   axis="columns")

# Drop some columns we now don't care about
df = df.drop(columns=["test_accuracy", "test_neuron_update_time", 
                      "test_presynaptic_update_time", "test_custom_update_reset_time",
                      "test_custom_update_gradient_batch_reduce_time", "test_custom_update_gradient_learn_time",
                      "test_custom_update_batch_softmax_1_time", "test_custom_update_batch_softmax_2_time",
                      "test_custom_update_batch_softmax_3_time", "test_custom_update_spike_count_reduce_time",
                      "test_custom_update_zero_out_post_time"])

df["time_per_timestep"] = df["test_gpu_time"] / (num_examples * 1000)
df["total_inference_energy"] = df["test_time"] * sim_power
df["total_sim_energy"] = df["test_gpu_time"] * (sim_power - idle_power)
df["sim_energy_per_example"] = df["total_sim_energy"] / num_examples
df["sim_energy_per_sop"] = df["total_sim_energy"] / num_sop

# Group by bar group params and aggregate across repeats
agg_df = df.groupby(["num_hidden"], as_index=False, dropna=False)
agg_df = agg_df.agg(mean_time_per_timestep=NamedAgg(column="time_per_timestep", aggfunc="mean"),
                    std_time_per_timestep=NamedAgg(column="time_per_timestep", aggfunc="std"),
                    mean_sim_energy_per_example=NamedAgg(column="sim_energy_per_example", aggfunc="mean"),
                    std_sim_energy_per_example=NamedAgg(column="sim_energy_per_example", aggfunc="std"),
                    mean_sim_energy_per_sop=NamedAgg(column="sim_energy_per_sop", aggfunc="mean"),
                    std_sim_energy_per_sop=NamedAgg(column="sim_energy_per_sop", aggfunc="std"))

def plot_bars(df, axis, bar_x, jetson_column_prefix, scale, y_label):
    # Plot Jetson bars
    jetson_actor = axis.bar(bar_x, df[f"mean_{jetson_column_prefix}"] * scale,
                            yerr=df[f"std_{jetson_column_prefix}"] * scale, width=0.4)

    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    axis.set_ylabel(y_label)

    return jetson_actor


fig, axes = plt.subplots(3, sharex=True, frameon=False, figsize=(plot_settings.column_width, 3.0))

xticks, xtick_index = np.unique(agg_df["num_hidden"], return_inverse=True)

jetson_actor = plot_bars(agg_df, axes[0], xtick_index, "time_per_timestep", 1e6, "Time per\ntimestep [us]")
plot_bars(agg_df, axes[1], xtick_index, "sim_energy_per_example", 1e3, "Energy per\nexample [mJ]")
plot_bars(agg_df, axes[2], xtick_index, "sim_energy_per_sop", 1e9, "Energy per\nSOP [nJ]")

axes[-1].set_xlabel("Number of hidden neurons")
axes[-1].set_xticks(np.arange(len(xticks)) + 0.3)
axes[-1].set_xticklabels(xticks)

fig.align_ylabels(axes)
fig.legend([jetson_actor], ["Jetson Xavier NX"],
            loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, 0.1, 1.0, 1.0])

fig.savefig("jetson_performance.pdf")
    
plt.show()
