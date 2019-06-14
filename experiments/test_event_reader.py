import os

import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_dl
import nengo_loihi
from nengo_loihi.dvs import EventsReader
from nengo_loihi.inputs import DVSFileChipNode
import tensorflow as tf

import davis_tracking

datafile = 'dataset/retinaTest95.events'

reader = EventsReader(datafile)

events = reader.read_events(rel_time=True)

plt.figure(figsize=(14,8))
window = 0.01
times = (2., 2.5, 3.)
for i, t in enumerate(times):
    plt.subplot(1, len(times), i+1)
    ti0 = int((t - window) * 1e6)
    ti1 = int((t) * 1e6)

    image = np.zeros((180, 240))
    for e in (e for e in events if e.t >= ti0 and e.t <= ti1):
        image[e.y, e.x] += 1 if e.polarity else -1

    plt.imshow(image, vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 100
amp = 1 / max_rate

with nengo.Network() as net:
    net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear()
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([1100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    inp = DVSFileChipNode(filename=datafile, t_start=2, pool=(10, 10), channels_last=False)
    e0 = nengo.Ensemble(18*24, 1)
    e1 = nengo.Ensemble(18*24, 1)

    w = 2
    nengo.Connection(inp[:18*24], e0.neurons, synapse=0.01,
                     transform=nengo.Convolution(
        n_filters=1, input_shape=(1, 18, 24), channels_last=False,
        kernel_size=(1, 1), init=nengo.dists.Choice([w])))
    nengo.Connection(inp[18*24:], e1.neurons, synapse=0.01,
                     transform=nengo.Convolution(
        n_filters=1, input_shape=(1, 18, 24), channels_last=False,
        kernel_size=(1, 1), init=nengo.dists.Choice([w])))

    p0 = nengo.Probe(e0.neurons, synapse=0.001)
    p1 = nengo.Probe(e1.neurons, synapse=0.001)


with nengo_loihi.Simulator(net) as sim:
    sim.run(1.0)

d0 = sim.data[p0]
d1 = sim.data[p1]
print((d0.min(), d0.max()))
print(d0[-1])

N = 3
plt.figure(figsize=(14,8))
indices = np.linspace(100, len(d0)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    # image = (d0[index]).reshape(18, 24) * 0.1
    image = (d0[index] - d1[index]).reshape(18, 24) * 0.01

    plt.imshow(image, vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])

plt.show()
