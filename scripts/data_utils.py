import os
import numpy as np

from pandas import DataFrame
from pandas.errors import ParserError

from collections import defaultdict
from glob import glob
from json import load
from pandas import concat, read_csv

def get_param_val(param_val):
    if isinstance(param_val, list):
        param_val = "_".join(str(p) for p in param_val)
    
    return param_val

def load_training_data_frame(keys, params_filter_fn=None, path="results"):
    # Build dictionary to hold data
    data = []

    # Loop through parameter files
    for name in glob(os.path.join(path, "params_*.json")):
        # Load parameters
        with open(name) as fp:
            params = load(fp)
        
        if params_filter_fn is None or params_filter_fn(params):
            # Get title from file
            title = os.path.splitext(os.path.basename(name))[0]

            train_filename = os.path.join(path, f"train_output_{title[7:]}.csv")
            if not os.path.exists(train_filename):
                print(f"ERROR: missing '{train_filename}'")
            else:
                # Load training data and calculate accuracy
                train_data = read_csv(train_filename, delimiter=",")
                train_data["Accuracy"] = 100.0 * train_data["Number correct"] / train_data["Num trials"]
                
                # Add columns to data frame and add resultant dataframe to list
                new_cols = {k: (get_param_val(params[k]) if k in params
                                else None) for k in keys}
                train_data = train_data.assign(**new_cols)
                data.append(train_data)

    # Build dataframe from dictionary and save
    return concat(data, ignore_index=True)

def load_data_test(data, test_filename, accuracy_key, time_key):
    if not os.path.exists(test_filename):
        print(f"ERROR: missing '{test_filename}'")
        data[accuracy_key].append(None)
        data[time_key].append(None)
    else:
        try:
            test_data = read_csv(test_filename, delimiter=",")
            assert len(test_data) == 1
            last_epoch_test_data = test_data.iloc[0]
            data[accuracy_key].append((100.0 * (last_epoch_test_data["Number correct"] / last_epoch_test_data["Num trials"])))
            data[time_key].append(test_data["Time"].sum())
        except ParserError as ex:
            print(f"ERROR: unable to parse '{test_filename}: {str(ex)}'")
            data[accuracy_key].append(None)
            data[time_key].append(None)
    
def load_data_frame(keys, params_filter_fn=None, path="results",
                    load_train=False, load_test=False, load_lava=False,
                    load_train_perf=False, load_test_perf=False):
    # Build dictionary to hold data
    data = defaultdict(list)

    # Loop through parameter files
    for name in glob(os.path.join(path, "params_*.json")):
        # Load parameters
        with open(name) as fp:
            params = load(fp)
        
        perf_keys = ["neuron_update", "presynaptic_update", 
                     "synapse_dynamics", "custom_update_deep_r_1", "custom_update_deep_r_2", "custom_update_deep_r_l1",
                     "custom_update_gradient_batch_reduce",
                     "custom_update_gradient_learn", "custom_update_reset", "custom_update_softmax_1","custom_update_softmax_2"]
        if params_filter_fn is None or params_filter_fn(params):            
            # Get title from file
            title = os.path.splitext(os.path.basename(name))[0]

            if load_train:
                assert "num_epochs" in params
                train_filename = os.path.join(path, f"train_output_{title[7:]}.csv")
                if not os.path.exists(train_filename):
                    print(f"ERROR: missing '{train_filename}'")
                    data["train_accuracy"].append(None)
                    data["train_time"].append(None)
                else:
                    try:
                        train_data = read_csv(train_filename, delimiter=",")

                        last_epoch_train_data = train_data[train_data["Epoch"] == (params["num_epochs"] - 1)]
                        if last_epoch_train_data.shape[0] == 1:
                            data["train_accuracy"].append((100.0 * (last_epoch_train_data["Number correct"] / last_epoch_train_data["Num trials"])).iloc[0])
                            data["train_time"].append(train_data["Time"].sum())
                        else:
                            print(f"ERROR: incomplete training data '{train_filename}'")
                            data["train_accuracy"].append(None)
                            data["train_time"].append(None)
                    except ParserError as ex:
                        print(f"ERROR: unable to parse '{train_filename}: {str(ex)}'")
                        data["train_accuracy"].append(None)
                        data["train_time"].append(None)
            
            if load_test:
                load_data_test(data, os.path.join(path, f"test_output_{title[7:]}.csv"),
                               "test_accuracy", "test_time")
            if load_lava:
                load_data_test(data, os.path.join(path, f"test_lava_output_{title[7:]}.csv"),
                               "test_lava_accuracy", "test_lava_time")
        
            if load_train_perf:
                # Load performance log
                with open(os.path.join(path, f"train_kernel_profile_{title[7:]}.json")) as fp:
                    perf = load(fp)
                
                # Add performance numbers to data
                for p in perf_keys:
                    data[f"train_{p}_time"].append(perf[p] if p in perf else None)

            if load_test_perf:
                # Load performance log
                with open(os.path.join(path, f"test_kernel_profile_{title[7:]}.json")) as fp:
                    perf = load(fp)
                
                # Add performance numbers to data
                for p in perf_keys:
                    data[f"test_{p}_time"].append(perf[p] if p in perf else None)

            # Add parameters to dictionary
            for k in keys:
                if k in params:
                    data[k].append(get_param_val(params[k]))
                else:
                    data[k].append(None)

    # Build dataframe from dictionary and save
    return DataFrame(data=data)
