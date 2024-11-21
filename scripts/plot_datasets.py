import matplotlib.pyplot as plt
import seaborn as sns

from tonic.datasets import SHD, SSC
from tonic.transforms import CropTime

from plot_settings import column_width
from random import choice

# Get SHD dataset, cropped to maximum timesteps (in us)
transform = CropTime(max=1E6)

# Create datasets
shd = SHD(save_to="../classifier/data", train=True, transform=transform)
ssc = SSC(save_to="../classifier/data", split="train",transform=transform)

# Read examples
shd_events, shd_label = choice(shd)
ssc_events, ssc_label = choice(ssc)

print(f"SHD class {shd.classes[shd_label]}")
print(f"SSC class {ssc.classes[ssc_label]}")
fig, axes = plt.subplots(1, 2, sharey=True, frameon=False, figsize=(column_width, 1.75))

axes[0].scatter(shd_events["t"] / 1000.0, shd_events["x"], s=0.5)
axes[1].scatter(ssc_events["t"] / 1000.0, ssc_events["x"], s=0.5)

axes[0].set_ylabel("Neuron ID")
axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

for a in axes:
    sns.despine(ax=a)
    a.xaxis.grid(False)
    a.yaxis.grid(False)
    a.set_xlabel("Time [ms]")
    a.set_xticks([0, 250, 500, 750, 1000])

fig.tight_layout(pad=0)
fig.savefig("datasets.pdf")
plt.show()

