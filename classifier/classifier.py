import csv
import numpy as np
import matplotlib.pyplot as plt
import lava.lib.dl.netx as netx
import logging

from argparse import ArgumentParser
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SourceRingBuffer
from lava.proc.monitor.process import Monitor
from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Callback, Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD
from tonic.transforms import CropTime, ToFrame

from hashlib import md5
from json import dump
from ml_genn.utils.data import preprocess_tonic_spikes
from ml_genn_netx import export
from random import choice
from tqdm import tqdm

from ml_genn.compilers.event_prop_compiler import default_params

BATCH_SIZE = 32
NUM_TEST_SAMPLES = 200


class EaseInSchedule(Callback):
    def set_params(self, compiled_network, **kwargs):
        self._optimisers = [o for o, _ in compiled_network.optimisers]

    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        for o in self._optimisers:
            if o.alpha < 0.001 :
                o.alpha = (0.001 / 1000.0) * (1.05 ** batch)
            else:
                o.alpha = 0.001


class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()


class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Copy events and shift in space by random amount
        events_copy = events.copy()
        events_copy["x"] += np.random.randint(-self.f_shift, self.f_shift)

        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy


class Blend:
    def __init__(self, p_blend, sensor_size, max_time, n_blend=7644):
        self.p_blend = p_blend
        self.sensor_size = sensor_size
        self.max_time = max_time
        self.n_blend = n_blend
        

    def __call__(self, dataset: list, classes: list) -> list:
        # Start with (shallow) copy of original dataset
        blended_dataset = dataset.copy()

        # Loop through number of blends to add
        for i in range(self.n_blend):
            # Pick random example
            idx = np.random.randint(0, len(dataset))
            example_spikes, example_label = dataset[idx]
            
            # Pick another from same class
            idx2 = np.random.randint(0, len(classes[example_label]))
            blend_spikes, blend_label = dataset[classes[example_label][idx2]]
            assert blend_label == example_label
            
            # Blend together to form new dataset
            blended_dataset.append((self.blend(example_spikes, blend_spikes),
                                    example_label))

        return blended_dataset

    def blend(self, X1, X2):
        # Copy spike arrays and align centres of mass in space and time
        X1 = X1.copy()
        X2 = X2.copy()
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)

        # Delete any spikes that are out of bounds in space or time
        X1 = np.delete(
            X1, np.where((X1["x"] < 0) | (X1["x"] >= self.sensor_size[0])
                         | (X1["t"] < 0) | (X1["t"] >= self.max_time)))
        X2 = np.delete(
            X2, np.where((X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) 
                         | (X2["t"] < 0) | (X2["t"] >= self.max_time)))

        # Combine random blended subset of spikes
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < (1.0 - self.p_blend)
        X1_X2 = np.concatenate((X1[mask1], X2[mask2]))

        # Resort and return
        idx = np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2

def load_data(train, dt, num_timesteps, num=None):
    # Get SHD dataset, cropped to maximum timesteps (in us)
    dataset = SHD(save_to="./data", train=train,
                  transform=CropTime(max=num_timesteps * dt * 1000.0))

    # Get raw event data
    raw_data = []
    for i in range(num if num is not None else len(dataset)):
        raw_data.append(dataset[i])

    return raw_data, dataset.sensor_size, dataset.ordering, len(dataset.classes)

def build_ml_genn_model(sensor_size, num_classes, num_hidden):
    network = Network(default_params)
    with network:
        # Populations
        input = Population(SpikeInput(max_spikes=BATCH_SIZE * 15000),
                           int(np.prod(sensor_size)), 
                           name="Input")
        hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                               tau_refrac=None),
                            num_hidden, name="Hidden", record_spikes=True)
        output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                            num_classes, name="Output")

        # Connections
        input_hidden = Connection(input, hidden, 
                                  Dense(Normal(mean=0.03, sd=0.01)),
                                  Exponential(5.0))
        Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
                   Exponential(5.0))
        Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
                   Exponential(5.0))

    return network, input, hidden, output, input_hidden

def train_genn(raw_dataset, network, serialiser, unique_suffix,
               input, hidden, output, input_hidden,
               sensor_size, ordering, num_epochs, dt, num_timesteps, reg_lambda):
    # Create EventProp compiler
    compiler = EventPropCompiler(example_timesteps=num_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 reg_lambda_upper=reg_lambda, reg_lambda_lower=reg_lambda, 
                                 reg_nu_upper=14, max_spikes=1500, 
                                 optimiser=Adam(0.001 / 1000.0), batch_size=BATCH_SIZE)
    # Create augmentation objects
    shift = Shift(40.0, sensor_size)
    blend = Blend(0.5, sensor_size, num_timesteps * dt * 1000.0)

    # Build classes list
    num_output = np.prod(output.shape)
    classes = [[] for _ in range(num_output)]
    for i, (_, label) in enumerate(raw_dataset):
        classes[label].append(i)
    
    # Compile network
    compiled_net = compiler.compile(network, name=f"classifier_train_{unique_suffix}")
    input_hidden_sg = compiled_net.connection_populations[input_hidden]
    
    # Train
    num_hidden = np.prod(hidden.shape)
    with compiled_net:
        callbacks = [Checkpoint(serialiser), EaseInSchedule(),
                     CSVTrainLog(f"train_output_{unique_suffix}.csv", output),
                     SpikeRecorder(hidden, key="hidden_spikes",
                                   record_counts=True)]
        # Loop through epochs
        for e in range(num_epochs):
            # Apply augmentation to events and preprocess
            spikes_train = []
            labels_train = []
            blended_dataset = blend(raw_dataset, classes)
            for events, label in blended_dataset:
                spikes_train.append(preprocess_tonic_spikes(shift(events), ordering,
                                                            sensor_size, dt=dt,
                                                            histogram_thresh=1))
                labels_train.append(label)

            # Train epoch
            metrics, cb_data  = compiled_net.train(
                {input: spikes_train}, {output: labels_train},
                start_epoch=e, num_epochs=1, 
                shuffle=True, callbacks=callbacks)

            # Sum number of hidden spikes in each batch
            hidden_spikes = np.zeros(num_hidden)
            for cb_d in cb_data["hidden_spikes"]:
                hidden_spikes += cb_d

            num_silent = np.count_nonzero(hidden_spikes==0)
            print(f"GeNN training epoch: {e}, Silent neurons: {num_silent}, Training accuracy: {100 * metrics[output].result}%")
            
            if num_silent > 0:
                input_hidden_sg.vars["g"].pull_from_device()
                g_view = input_hidden_sg.vars["g"].view.reshape((np.prod(input.shape), num_hidden))
                g_view[:,hidden_spikes==0] += 0.002
                input_hidden_sg.vars["g"].push_to_device()

def evaluate_genn(raw_dataset, network, unique_suffix,
                  input, hidden, output, 
                  sensor_size, ordering, plot, 
                  dt, num_timesteps):
    # Preprocess
    spikes = []
    labels = []
    for events, label in raw_dataset:
        spikes.append(preprocess_tonic_spikes(events, ordering,
                                              sensor_size, dt=dt,
                                              histogram_thresh=1))
        labels.append(label)
    
    compiler = InferenceCompiler(evaluate_timesteps=num_timesteps,
                                 reset_in_syn_between_batches=True,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network, name=f"classifier_test_{unique_suffix}")

    with compiled_net:
        callbacks = ["batch_progress_bar"]
        if plot:
            callbacks.extend([SpikeRecorder(hidden, key="hidden_spikes"),
                              VarRecorder(output, "v", key="output_v")])

        metrics, cb_data  = compiled_net.evaluate({input: spikes},
                                                  {output: labels},
                                                  callbacks=callbacks)

        print(f"GeNN test accuracy: {100 * metrics[output].result}%")
        
        if plot:
            fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row")
            for a in range(NUM_TEST_SAMPLES):
                axes[0, a].scatter(cb_data["hidden_spikes"][0][a], cb_data["hidden_spikes"][1][a], s=1)
                axes[1, a].plot(cb_data["output_v"][a])
            
            axes[0, 0].set_ylabel("Hidden neuron ID")
            axes[1, 0].set_ylabel("Output voltage")

def evaluate_lava(raw_dataset, net_x_filename, 
                  sensor_size, num_classes, plot, 
                  num_timesteps):
    # Preprocess
    num_input = int(np.prod(sensor_size))
    transform = ToFrame(sensor_size=sensor_size, time_window=1000.0)
    tensors = []
    labels = []
    for events, label in raw_dataset:
        # Transform events to tensor
        tensor = transform(events)
        assert tensor.shape[-1] < num_timesteps

        # Transpose tensor and pad time to max
        tensors.append(np.pad(np.reshape(np.transpose(tensor), (num_input, -1)),
                              ((0, 0), (0, num_timesteps - tensor.shape[0]))))
        labels.append(label)

    # Stack tensors
    tensors = np.hstack(tensors)

    network_lava = netx.hdf5.Network(net_config=net_x_filename, reset_interval=num_timesteps)

    # **TODO** move to recurrent unit test
    assert network_lava.input_shape == (num_input,)
    assert len(network_lava) == 2
    assert type(network_lava.layers[0]) == netx.blocks.process.RecurrentDense
    assert type(network_lava.layers[1]) == netx.blocks.process.Dense

    # Create source ring buffer to deliver input spike tensors and connect to network input port
    input_lava = SourceRingBuffer(data=tensors)
    input_lava.s_out.connect(network_lava.inp)

    # Create monitor to record output voltages (shape is total timesteps)
    monitor_output = Monitor()
    monitor_output.probe(network_lava.layers[-1].neuron.v, NUM_TEST_SAMPLES * num_timesteps)

    if plot:
        monitor_hidden = Monitor()
        monitor_hidden.probe(network_lava.layers[0].neuron.s_out, NUM_TEST_SAMPLES * num_timesteps)

    run_config = Loihi2SimCfg(select_tag="fixed_pt")

    # Run model for each test sample
    for _ in tqdm(range(NUM_TEST_SAMPLES)):
        network_lava.run(condition=RunSteps(num_steps=num_timesteps), run_cfg=run_config)

    # Get output and reshape
    output_v = monitor_output.get_data()["neuron"]["v"]
    output_v = np.reshape(output_v, (NUM_TEST_SAMPLES, MAX_TIMESTEPS, num_classes))

    # Calculate output weighting
    output_weighting = np.exp(-np.arange(num_timesteps) / num_timesteps)

    # For each example, sum weighted output neuron voltage over time
    sum_v = np.sum(output_v * output_weighting[np.newaxis,:,np.newaxis], axis=1)

    # Find maximum output neuron voltage and compare to label
    pred = np.argmax(sum_v, axis=1)
    good = np.sum(pred == labels)

    print(f"Lava test accuracy: {good/NUM_TEST_SAMPLES*100}%")
    if plot:
        hidden_spikes = monitor_hidden.get_data()["neuron"]["s_out"]
        hidden_spikes = np.reshape(hidden_spikes, (NUM_TEST_SAMPLES, num_timesteps, num_hidden))
        
        fig, axes = plt.subplots(2, NUM_TEST_SAMPLES, sharex="col", sharey="row")
        for a in range(NUM_TEST_SAMPLES):
            sample_hidden_spikes = np.where(hidden_spikes[a,:,:] > 0.0)
            axes[0, a].scatter(sample_hidden_spikes[0], sample_hidden_spikes[1], s=1)
            axes[1, a].plot(output_v[a,:,:])
        
        axes[0,0].set_ylabel("Hidden neuron ID")
        axes[1,0].set_ylabel("Output voltage")

    network_lava.stop()


parser = ArgumentParser()
parser.add_argument("--mode", choices=["train", "test_genn", "test_lava", "test_loihi"], default="train")
parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
parser.add_argument("--plot", action="store_true", help="Plot debug")
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
#parser.add_argument("--dataset", choices=["ssc", "shd"], required=True)
parser.add_argument("--num-hidden", type=int, help="Number of hidden neurons")
parser.add_argument("--reg-lambda", type=float, help="EventProp regularization strength")
parser.add_argument("--dt", type=float, help="Simulation timestep")
parser.add_argument("--num-timesteps", type=int, required=True, help="Number of simulation timesteps")

args = parser.parse_args()

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items()
                         if arg not in ["mode", "kernel_profiling"])
                         

# Get SHD data
if args.mode == "train":
    raw_train_data, sensor_size, ordering, num_classes = load_data(True, args.dt, args.num_timesteps)
    raw_test_data, _, _, _ = load_data(False, args.dt, args.num_timesteps, NUM_TEST_SAMPLES)
else:
    raw_test_data, sensor_size, ordering, num_classes = load_data(False, args.dt, args.num_timesteps, NUM_TEST_SAMPLES)

# Build suitable mlGeNN model
network, input, hidden, output, input_hidden = build_ml_genn_model(sensor_size, num_classes, args.num_hidden)

serialiser = Numpy(f"checkpoints_{unique_suffix}")

if args.mode == "train":
    train_genn(raw_train_data, network, serialiser, unique_suffix,
                input, hidden, output, input_hidden,
                sensor_size, ordering, args.num_epochs, 
                args.dt, args.num_timesteps, args.reg_lambda)
    
# Load checkpoints and export to NETX
network.load((args.num_epochs - 1,), serialiser)

export(f"shd_{unique_suffix}.net", input, output, dt=args.dt)


if args.mode == "test_genn":
    evaluate_genn(raw_test_data, network, unique_suffix,
                  input, hidden, output, 
                  sensor_size, ordering,
                  args.dt, args.num_timesteps, args.plot)
elif args.mode == "test_lava" or args.mode == "test_loihi":
    evaluate_lava(raw_test_data, f"shd_{unique_suffix}.net", sensor_size, num_classes, 
                  args.plot, args.num_timesteps)

if args.plot:
    plt.show()







