import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs

from itertools import chain
from pandas import read_csv

import plot_settings

data = read_csv("spyx_genn.csv", delimiter=",")


data = data.sort_values(by="Timestep [ms]", ascending=False)

data_1024_hidden = data[data["Num hidden"] == 1024]
data_256_hidden = data[data["Num hidden"] == 256]

fig, axes = plt.subplots(2, sharex=True, frameon=False, figsize=(plot_settings.column_width, 3.0))


# Plot memory
actor_1024 = axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
                          data_1024_hidden["Thomas peak GPU [mb]"] - data_1024_hidden["Thomas dataset GPU [mb]"],
                          marker="o")
actor_256 = axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
                         data_256_hidden["Thomas peak GPU [mb]"] - data_256_hidden["Thomas dataset GPU [mb]"], 
                         marker="o")
axes[0].plot(1000.0 / data_1024_hidden["Timestep [ms]"], 
             data_1024_hidden["Spyx platform peak GPU [mb]"] - data_1024_hidden["Spyx dataset GPU [mb]"],
             marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[0].plot(1000.0 / data_256_hidden["Timestep [ms]"], 
             data_256_hidden["Spyx platform peak GPU [mb]"] - data_256_hidden["Spyx dataset GPU [mb]"],
             marker="o", color=actor_256[0].get_color(), linestyle="--")


axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Thomas time [s]"] + data_1024_hidden["Thomas build load time [s]"], marker="o", color=actor_1024[0].get_color())
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Thomas time [s]"] + data_256_hidden["Thomas build load time [s]"], marker="o", color=actor_256[0].get_color())
axes[1].plot(1000.0 / data_1024_hidden["Timestep [ms]"], data_1024_hidden["Spyx time default [s]"], marker="o", color=actor_1024[0].get_color(), linestyle="--")
axes[1].plot(1000.0 / data_256_hidden["Timestep [ms]"], data_256_hidden["Spyx time default [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[0].set_ylabel("GPU memory [MiB]")
axes[1].set_ylabel("Training time [s]")
axes[0].set_ylim((0, 3000))
axes[1].set_ylim((0, 4000))

axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

for a in axes[:2]:
    a.set_xlabel("Num timesteps")
    a.grid(axis="y")
    a.grid(which='minor', alpha=0.3)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

fig.legend([actor_256[0],  mlines.Line2D([],[], color="black"), actor_1024[0],mlines.Line2D([],[], linestyle="--", color="black")], 
           ["256 hidden neurons", "GeNN", "1024 hidden neurons",  "Spyx"], 
           loc="lower center", ncol=2, frameon=False)

fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])


fig.savefig("spyx_genn_benchmark.pdf")

plt.show()
