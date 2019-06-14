import os

import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_dl
import nengo_loihi
from nengo_loihi.inputs import DVSFileChipNode
import tensorflow as tf

import davis_tracking


datafile = 'dataset/retinaTest95.events'

times, images, targets = davis_tracking.load_data(
    datafile,
    dt=0.1,                  # time step between images
    decay_time=0.01,         # low pass filter time constant
    separate_channels=True,  # do positive and negative spikes separately
    saturation=10,           # clip values to this range
    merge=10)                # merge pixels together to reduce size

if 0:
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

    plt.figure()
    plt.plot(times, targets)
    plt.legend(['x', 'y', 'radius', 'present'])

    plt.show()

# just care about x-y coords
targets = targets[:, :2]


input_shape = (2, 18, 24)
dimensions = input_shape[0]*input_shape[1]*input_shape[2]

max_rate = 500
amp = 1 / max_rate

with nengo.Network() as net:

    inp = nengo.Node([0]*dimensions)

    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    convnet = davis_tracking.ConvNet(
        net=nengo.Network(label='convnet'),
        max_rate=max_rate)
    convnet.input = inp

    # config after convnet to override default values
    convnet.net.config[nengo.Ensemble].neuron_type = (
        nengo.SpikingRectifiedLinear(amplitude=amp))

    convnet.make_input_layer(input_shape,
                             spatial_size=(12,12),
                             spatial_stride=(6,6))
    # nengo.Connection(inp, convnet.input)

    convnet.make_middle_layer(n_features=10, n_parallel=6, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3, 3))

    convnet.make_middle_layer(n_features=15, n_parallel=4, n_local=3,
                              kernel_stride=(1,1), kernel_size=(3, 3))

    convnet.make_output_layer(2)

    out = nengo.Node(None, size_in=2)
    nengo.Connection(convnet.output, out)

    p_out = nengo.Probe(out)


print([ens.n_neurons for ens in net.all_ensembles])


inputs_train = images[::2]
inputs_test = images[1::2]
targets_train = targets[::2]
targets_test = targets[1::2]

minibatch_size = 200
# n_epochs = 5
n_epochs = 100
# n_epochs = 400
learning_rate = 1e-4

loadfile = None
# loadfile = 'loihi_splitnet.params'

N = len(inputs_train)
n_steps = int(np.ceil(N/minibatch_size))
dl_train_data = {inp: np.resize(inputs_train, (minibatch_size, n_steps, dimensions)),
                 p_out: np.resize(targets_train, (minibatch_size, n_steps, 2))}
N = len(inputs_test)
n_steps = int(np.ceil(N/minibatch_size))
dl_test_data = {inp: np.resize(inputs_test, (minibatch_size, n_steps, dimensions)),
                p_out: np.resize(targets_test, (minibatch_size, n_steps, 2))}
with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:

    if loadfile and os.path.exists(loadfile):
        sim.load_params(loadfile)
        sim.freeze_params(net)
    else:
        loss_pre = sim.loss(dl_test_data)
        print('loss pre:', loss_pre)

        sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=learning_rate),
              n_epochs=n_epochs)

        loss_post = sim.loss(dl_test_data)
        print('loss post:', loss_post)

        sim.run_steps(n_steps, data=dl_test_data)

        # store trained parameters back into the network
        sim.freeze_params(net)
        # if loadfile:
            # sim.save_params(loadfile)

        data = sim.data[p_out].reshape(-1,2)[:len(targets_test)]
        if 0:
            plt.figure()
            plt.plot(times[1::2], data*10)
            plt.plot(times[1::2], targets_test*10)
            plt.show()

        rmse = np.sqrt(np.mean((data-targets_test)**2, axis=0))*10
        print(rmse)


t_start = times[0]
with net:
    inp2 = DVSFileChipNode(filename=datafile, t_start=t_start, pool=(10, 10))

    for conn in convnet.net.connections[:]:
        if conn.pre_obj is inp:
            convnet.net.connections.remove(conn)
            nengo.Connection(inp2[conn.pre_slice], conn.post,
                             transform=conn.transform,
            )

    net.nodes.remove(inp)


with nengo_loihi.Simulator(net) as sim:
    sim.run(1.5)

plt.plot(times, targets)
plt.plot(sim.trange() + t_start, sim.data[p_out])
plt.show()
