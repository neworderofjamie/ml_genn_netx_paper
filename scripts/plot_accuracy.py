import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width

def plot_accuracy_axis(df, axis, bar_group_params):
    # Group by bar group params and aggregate across repeats
    group_df = df.groupby(bar_group_params, as_index=False, dropna=False)
    group_df = group_df.agg(mean_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="mean"),
                            std_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="std"),
                            mean_test_lava_accuracy=NamedAgg(column="test_lava_accuracy", aggfunc="mean"),
                            std_test_lava_accuracy=NamedAgg(column="test_lava_accuracy", aggfunc="std"),
                            mean_train_accuracy=NamedAgg(column="train_accuracy", aggfunc="mean"),
                            std_train_accuracy=NamedAgg(column="train_accuracy", aggfunc="std"))

    # Find unique hidden sizes and their indices
    xticks, xtick_index = np.unique(group_df["num_hidden"], return_inverse=True)

    train_actor = axis.bar(xtick_index, group_df["mean_train_accuracy"],
                           yerr=group_df["std_train_accuracy"], width=0.2)
    genn_test_actor = axis.bar(xtick_index + 0.2, group_df["mean_test_accuracy"],
                               yerr=group_df["std_test_accuracy"], width=0.2)
    lava_test_actor = axis.bar(xtick_index + 0.4, group_df["mean_test_lava_accuracy"],
                               yerr=group_df["std_test_lava_accuracy"], width=0.2)
  

    sns.despine(ax=axis)
    ax.xaxis.grid(False)
    axis.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    axis.set_xticks(np.arange(len(xticks)) + 0.3)
    
    axis.set_xticklabels(xticks)

    return [train_actor, genn_test_actor, lava_test_actor]

axis_group_params = ["dataset"]
bar_group_params = ["num_hidden"]

keys = bar_group_params + axis_group_params
df = load_data_frame(keys, lambda p: p["dt"] == 1.0,
                     path=os.path.join("..", "classifier"),
                     load_train=True, load_test=True, load_lava=True)
    
axes_df = df.groupby(axis_group_params, as_index=False, dropna=False)
fig, axes = plt.subplots(1, len(axes_df), sharey=True, figsize=(column_width, 1.75))
for (name, ax_df), ax in zip(axes_df, axes):
    if isinstance(name, str):
        ax.set_title(name.upper())
    else:
        ax.set_title(name[0].upper())
    ax.set_xlabel("Hidden layer size")
    actors = plot_accuracy_axis(ax_df, ax, bar_group_params)
    
axes[0].set_ylabel("Accuracy [%]")

fig.legend(actors, ["mlGeNN train", "mlGeNN test", "Lava fixed-point test", "Loihi test"],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, 0.225, 1.0, 1.0])

fig.savefig("accuracy.pdf")

plt.show()
