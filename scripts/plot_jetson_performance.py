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

for b in [None, 32, 64, 128, 256]:
    print(f"BATCH SIZE {b}")
    # Load dataframe       
    keys = ["dataset", "num_hidden", "inference_batch_size"]
    df = load_data_frame(keys, lambda p: p["dt"] == 1.0 and p["augmentation"] != "plain" and p["seed"] == 1234 and p["dataset"] == "shd",
                        path=os.path.join("..", "classifier"),
                        load_test=True, load_test_perf=True, load_jetson_power=True,
                        inference_batch_size=b)

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
                          "test_custom_update_zero_out_post_time", "dataset"])

    num_batches = num_examples if b is None else int(np.ceil(num_examples / b))
    df["latency"] = df["test_gpu_time"] / num_batches
    total_energy = df["test_gpu_time"] * df["jetson_sim_power"]
    total_dynamic_energy = df["test_gpu_time"] * (df["jetson_sim_power"] - df["jetson_idle_power"])
    df["total_energy_per_example"] = total_energy / num_examples
    df["dynamic_energy_per_example"] = total_dynamic_energy / num_examples

    df["total_energy_per_example"] *= 1e3
    df["dynamic_energy_per_example"] *= 1e3
    df["latency"] *= 1e3
    df["total_energy_delay_product"] = df["total_energy_per_example"] * df["latency"]
    print(df)
