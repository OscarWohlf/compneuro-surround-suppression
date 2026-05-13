import numpy as np
import matplotlib.pyplot as plt
import os
from src.lif import DT

def save_or_show(path=None):
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=200)
        plt.close()


def plot_raster(spikes, dt=DT, path=None):
    spike_steps, neuron_ids = np.where(spikes > 0)
    times_ms = spike_steps * dt

    plt.figure(figsize=(10, 6))
    plt.scatter(times_ms, neuron_ids, s=5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.title("Raster plot of spike times")
    save_or_show(path)


def plot_rate_and_input(times, rate_hz, input_current, path=None):
    input_scaled = input_current / np.max(input_current) * np.max(rate_hz)

    plt.figure(figsize=(10, 5))
    plt.plot(times, rate_hz, label="Mean firing rate")
    plt.plot(times, input_scaled, "--", label="External input, rescaled")
    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Mean firing rate compared with external input")
    plt.legend()
    save_or_show(path)