import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs
import numpy as np
import os
import seaborn as sns

from data_utils import load_data_frame
from itertools import chain
from pandas import read_csv

import plot_settings

genn_peak_memory = {(256, 1): 502, (256, 2): 470, (256, 5): 468,
                    (1024, 1): 972, (1024, 2): 970, (1024, 5): 968}

# Load new GeNN data
genn_df = load_data_frame(["dt", "num_hidden"], lambda p: p["augmentation"] == "plain" and p["seed"] == 1234,
                          path=os.path.join("..", "classifier"),
                          load_train=True, load_train_perf=True)

# Calculate total GPU training time
genn_df["train_gpu_time"] = (genn_df["train_neuron_update_time"] + genn_df["train_presynaptic_update_time"] + genn_df["train_custom_update_reset_time"]
                             + genn_df["train_custom_update_gradient_batch_reduce_time"] + genn_df["train_custom_update_gradient_learn_time"]
                             + genn_df["train_custom_update_batch_softmax_1_time"] + genn_df["train_custom_update_batch_softmax_2_time"] 
                             + genn_df["train_custom_update_batch_softmax_3_time"] + genn_df["train_custom_update_spike_count_reduce_time"]
                             + genn_df["train_custom_update_zero_out_post_time"])

# Load old GeNN and Spyx data and drop GeNN columns
spyx_df = read_csv("spyx_genn.csv", delimiter=",")
spyx_df = spyx_df.drop(columns=["Thomas time [s]", "Thomas peak GPU [mb]", "Thomas dataset GPU [mb]",
                                "Thomas neuron time [s]", "Thomas neuron time [s]"])

# Join and drop duplicate columns
data = spyx_df.merge(genn_df, "left", left_on=["Num hidden", "Timestep [ms]"], right_on=["num_hidden", "dt"])
data = data.drop(columns=["num_hidden", "dt"])

data = data.sort_values(by="Timestep [ms]", ascending=False)

data["genn_peak_memory"] = data.apply(lambda f: genn_peak_memory[(f["Num hidden"], f["Timestep [ms]"])],
                                      axis="columns")

data_1024_hidden = data[data["Num hidden"] == 1024]
data_256_hidden = data[data["Num hidden"] == 256]

fig, axes = plt.subplots(1, 2, frameon=False, figsize=(plot_settings.column_width, 1.75))


# Plot memory
actor_1024 = axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                          data_1024_hidden["genn_peak_memory"],
                          marker="o")
actor_256 = axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                         data_256_hidden["genn_peak_memory"],
                         marker="o")
axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
             data_1024_hidden["Spyx platform peak GPU [mb]"] - data_1024_hidden["Spyx dataset GPU [mb]"],
             marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
             data_256_hidden["Spyx platform peak GPU [mb]"] - data_256_hidden["Spyx dataset GPU [mb]"],
             marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["train_gpu_time"] + data_1024_hidden["Thomas build load time [s]"], marker="o", color=actor_1024[0].get_color())
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["train_gpu_time"] + data_256_hidden["Thomas build load time [s]"], marker="o", color=actor_256[0].get_color())
axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Spyx time default [s]"], marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Spyx time default [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[0].set_ylabel("GPU memory [MiB]")
axes[1].set_ylabel("Training time [s]")
axes[0].set_ylim((0, 3000))
axes[1].set_ylim((0, 4000))
axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

for a in axes:
    a.set_xlabel("Num timesteps")
    sns.despine(ax=a)
    a.xaxis.grid(False)

fig.legend([actor_256[0],  mlines.Line2D([],[], color="black"), actor_1024[0],mlines.Line2D([],[], linestyle="--", color="black")], 
           ["256 hidden neurons", "mlGeNN", "1024 hidden neurons",  "Spyx"], 
           loc="lower center", ncol=2, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.225, 1.0, 1.0])


fig.savefig("spyx_genn_benchmark.pdf")

plt.show()
