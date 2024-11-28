import os
import plot_settings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pandas import NamedAgg

from data_utils import load_data_frame

num_examples = 2264
num_sops = {256: 7154159044, 512: 18815875280,
            1024: 53417193820}

# Load dataframe       
keys = ["dataset", "num_hidden"]
df = load_data_frame(keys, lambda p: p["dt"] == 1.0 and p["augmentation"] != "plain" and p["seed"] == 1234 and p["dataset"] == "shd",
                     path=os.path.join("..", "classifier"),
                     load_test=True, load_test_perf=True, load_jetson_power=True)

# Sort
df = df.sort_values(["dataset", "num_hidden"])

# Calculate new columns
df["test_gpu_time"] = df["test_neuron_update_time"] + df["test_presynaptic_update_time"] + df["test_custom_update_reset_time"]
num_sop = df["num_hidden"].map(num_sops)

print(df["test_gpu_time"] / df["test_time"])
# Drop some columns we now don't care about
df = df.drop(columns=["test_accuracy", "test_neuron_update_time", 
                      "test_presynaptic_update_time", "test_custom_update_reset_time",
                      "test_custom_update_gradient_batch_reduce_time", "test_custom_update_gradient_learn_time",
                      "test_custom_update_batch_softmax_1_time", "test_custom_update_batch_softmax_2_time",
                      "test_custom_update_batch_softmax_3_time", "test_custom_update_spike_count_reduce_time",
                      "test_custom_update_zero_out_post_time"])

df["time_per_timestep"] = df["test_gpu_time"] / (num_examples * 1000)
df["time_per_example"] = df["test_gpu_time"] / num_examples
df["total_inference_energy"] = df["test_time"] * df["jetson_sim_power"]
df["total_sim_energy"] = df["test_gpu_time"] * (df["jetson_sim_power"] - df["jetson_idle_power"])
df["inference_energy_per_example"] = df["total_inference_energy"] / num_examples
df["sim_energy_per_example"] = df["total_sim_energy"] / num_examples
df["sim_energy_per_sop"] = df["total_sim_energy"] / num_sop

df["inference_energy_per_example"] *= 1e3
df["sim_energy_per_example"] *= 1e3
df["time_per_example"] *= 1e3
df["inference_energy_delay_product"] = df["inference_energy_per_example"] * df["time_per_example"]
print(df)
def plot_bars(df, axis, bar_x, jetson_column_prefix, scale, y_label):
    # Plot Jetson bars
    jetson_actor = axis.bar(bar_x, df[jetson_column_prefix] * scale,
                            yerr=df[jetson_column_prefix] * scale, width=0.4)

    sns.despine(ax=axis)
    axis.xaxis.grid(False)
    axis.set_ylabel(y_label)

    return jetson_actor


fig, axes = plt.subplots(3, sharex=True, frameon=False, figsize=(plot_settings.column_width, 3.0))

xticks, xtick_index = np.unique(df["num_hidden"], return_inverse=True)

jetson_actor = plot_bars(df, axes[0], xtick_index, "time_per_timestep", 1e6, "Time per\ntimestep [us]")
plot_bars(df, axes[1], xtick_index, "sim_energy_per_example", 1e3, "Energy per\nexample [mJ]")
plot_bars(df, axes[2], xtick_index, "sim_energy_per_sop", 1e9, "Energy per\nSOP [nJ]")

axes[-1].set_xlabel("Number of hidden neurons")
axes[-1].set_xticks(np.arange(len(xticks)) + 0.3)
axes[-1].set_xticklabels(xticks)

fig.align_ylabels(axes)
fig.legend([jetson_actor], ["Jetson Orin Nano"],
            loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, 0.1, 1.0, 1.0])

fig.savefig("jetson_performance.pdf")
    
plt.show()
