import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import load_data_frame

idle_power = 6.1
power = {256: 7.4, 512: 7.5, 1024: 7.6}
dataset_num_examples = {"shd": 2264, "ssc": 20382}

# Load dataframe       
keys = ["dataset", "num_hidden"]
df = load_data_frame(keys, lambda p: p["dt"] == 1.0 and p["augmentation"] != "plain" and p["seed"] == 1234,
                     path=os.path.join("..", "classifier"),
                     load_test=True, load_test_perf=True)

df["test_gpu_time"] = df["test_neuron_update_time"] + df["test_presynaptic_update_time"] + df["test_custom_update_reset_time"]
num_examples = df["dataset"].map(dataset_num_examples)
sim_power = df["num_hidden"].map(power)

# Drop some tables we don't care about
df = df.drop(columns=["test_accuracy", "test_neuron_update_time", 
                      "test_presynaptic_update_time", "test_custom_update_reset_time",
                      "test_custom_update_gradient_batch_reduce_time", "test_custom_update_gradient_learn_time",
                      "test_custom_update_batch_softmax_1_time", "test_custom_update_batch_softmax_2_time",
                      "test_custom_update_batch_softmax_3_time", "test_custom_update_spike_count_reduce_time",
                      "test_custom_update_zero_out_post_time"])

df["time_per_timestep"] = 1e6 * (df["test_gpu_time"] / (num_examples * 1000))
df["total_inference_energy"] = df["test_time"] * sim_power
df["total_sim_energy"] = df["test_gpu_time"] * (sim_power - idle_power)
df["sim_energy_per_example"] = df["total_sim_energy"] / num_examples
print(df)
