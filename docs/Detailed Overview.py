#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_dl
import tensorflow as tf

import davis_tracking


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.01,         # low pass filter time constant
                                                  separate_channels=False, # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=1)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.1,          # low pass filter time constant
                                                  separate_channels=False, # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=1)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.005,          # low pass filter time constant
                                                  separate_channels=True,  # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=1)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axhline(images.shape[1]//2, c='k')
    plt.axhline(targets[index,1]+images.shape[1]//2, c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.1,          # low pass filter time constant
                                                  separate_channels=True,  # do positive and negative spikes separately
                                                  saturation=0.5,          # clip values to this range
                                                  merge=1)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axhline(images.shape[1]//2, c='k')
    plt.axhline(targets[index,1]+images.shape[1]//2, c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.1,          # low pass filter time constant
                                                  separate_channels=True,  # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=5)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axhline(images.shape[1]//2, c='k')
    plt.axhline(targets[index,1]+images.shape[1]//2, c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')


plt.plot(times, targets)
plt.legend(['x', 'y', 'radius', 'present']);


times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.01,                 # time step between images
                                                  decay_time=0.01,         # low pass filter time constant
                                                  separate_channels=True,  # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=10)                # merge pixels together to reduce size
targets = targets[:, :2]


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 100
amp = 1 / max_rate

model = nengo.Network()
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None

    inp = nengo.Node([0]*dimensions)

    convnet = davis_tracking.ConvNet(nengo.Network(label='convnet'))
    
    convnet.make_input_layer(input_shape)
    nengo.Connection(inp, convnet.input)
    
    convnet.make_middle_layer(n_features=60, n_parallel=1, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_middle_layer(n_features=60, n_parallel=1, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_output_layer(2)
    
    out = nengo.Node(None, size_in=2)
    nengo.Connection(convnet.output, out)

    p_out = nengo.Probe(out)


#import nengo_gui.jupyter
#nengo_gui.jupyter.InlineGUI(model)


inputs_train = images[::2]
inputs_test = images[1::2]
targets_train = targets[::2]
targets_test = targets[1::2]

minibatch_size = 200
n_epochs = 400
learning_rate = 1e-4


N = len(inputs_train)
n_steps = int(np.ceil(N/minibatch_size))
dl_train_data = {inp: np.resize(inputs_train, (minibatch_size, n_steps, dimensions)),
                 p_out: np.resize(targets_train, (minibatch_size, n_steps, 2))}
N = len(inputs_test)
n_steps = int(np.ceil(N/minibatch_size))
dl_test_data = {inp: np.resize(inputs_test, (minibatch_size, n_steps, dimensions)),
                p_out: np.resize(targets_test, (minibatch_size, n_steps, 2))}
with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
    loss_pre = sim.loss(dl_test_data)
    print('loss pre:', loss_pre)

    sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=learning_rate),
          n_epochs=n_epochs)

    loss_post = sim.loss(dl_test_data)
    print('loss post:', loss_post)

    sim.run_steps(n_steps, data=dl_test_data)


print(inputs_train.shape)


data = sim.data[p_out].reshape(-1,2)[:len(targets_test)]
plt.plot(times[1::2], data*10)
plt.plot(times[1::2], targets_test*10)

rmse = np.sqrt(np.mean((data-targets_test)**2, axis=0))*10
print(rmse)


print([ens.n_neurons for ens in model.all_ensembles])


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 100
amp = 1 / max_rate

model = nengo.Network()
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None

    inp = nengo.Node([0]*dimensions)

    convnet = davis_tracking.ConvNet(nengo.Network(label='convnet'))
    
    convnet.make_input_layer(input_shape)
    nengo.Connection(inp, convnet.input)
    
    convnet.make_middle_layer(n_features=20, n_parallel=3, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_middle_layer(n_features=30, n_parallel=2, n_local=2,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_output_layer(2)
    
    out = nengo.Node(None, size_in=2)
    nengo.Connection(convnet.output, out)

    p_out = nengo.Probe(out)


#import nengo_gui.jupyter
#nengo_gui.jupyter.InlineGUI(model)


print([ens.n_neurons for ens in model.all_ensembles])


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 100
amp = 1 / max_rate

model = nengo.Network()
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None

    inp = nengo.Node([0]*dimensions)

    convnet = davis_tracking.ConvNet(nengo.Network(label='convnet'))
    
    convnet.make_input_layer(input_shape)
    nengo.Connection(inp, convnet.input)
    
    convnet.make_middle_layer(n_features=2, n_parallel=30, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_middle_layer(n_features=3, n_parallel=20, n_local=2,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_output_layer(2)
    
    out = nengo.Node(None, size_in=2)
    nengo.Connection(convnet.output, out)

    p_out = nengo.Probe(out)
    
print([ens.n_neurons for ens in model.all_ensembles])


#import nengo_gui.jupyter
#nengo_gui.jupyter.InlineGUI(model)


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 100
amp = 1 / max_rate

model = nengo.Network()
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None

    inp = nengo.Node([0]*dimensions)

    convnet = davis_tracking.ConvNet(nengo.Network(label='convnet'))
    
    convnet.make_input_layer(input_shape,
                             spatial_size=(12,12),
                             spatial_stride=(6,6))
    nengo.Connection(inp, convnet.input)
    
    convnet.make_middle_layer(n_features=10, n_parallel=6, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_middle_layer(n_features=15, n_parallel=4, n_local=3,
                              kernel_stride=(1,1), kernel_size=(3, 3))
    
    convnet.make_output_layer(2)
    
    out = nengo.Node(None, size_in=2)
    nengo.Connection(convnet.output, out)

    p_out = nengo.Probe(out)
    
print([ens.n_neurons for ens in model.all_ensembles])


#import nengo_gui.jupyter
#nengo_gui.jupyter.InlineGUI(model)


inputs_train = images[::2]
inputs_test = images[1::2]
targets_train = targets[::2]
targets_test = targets[1::2]

minibatch_size = 200
n_epochs = 400
learning_rate = 1e-4


N = len(inputs_train)
n_steps = int(np.ceil(N/minibatch_size))
dl_train_data = {inp: np.resize(inputs_train, (minibatch_size, n_steps, dimensions)),
                 p_out: np.resize(targets_train, (minibatch_size, n_steps, 2))}
N = len(inputs_test)
n_steps = int(np.ceil(N/minibatch_size))
dl_test_data = {inp: np.resize(inputs_test, (minibatch_size, n_steps, dimensions)),
                p_out: np.resize(targets_test, (minibatch_size, n_steps, 2))}
with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
    loss_pre = sim.loss(dl_test_data)
    print('loss pre:', loss_pre)

    sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=learning_rate),
          n_epochs=n_epochs)

    loss_post = sim.loss(dl_test_data)
    print('loss post:', loss_post)

    sim.run_steps(n_steps, data=dl_test_data)


data = sim.data[p_out].reshape(-1,2)[:len(targets_test)]
plt.plot(times[1::2], data*10)
plt.plot(times[1::2], targets_test*10)

rmse = np.sqrt(np.mean((data-targets_test)**2, axis=0))*10
print(rmse)




