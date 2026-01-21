import nengo
import numpy as np

model = nengo.Network(label="Clock-style Counter")
with model:

    # ---- Input: a switch (0 or 1) ----
    def switch(t):
        # Turn "on" briefly every second
        #return 1.0 if t >= 2 and t <= 2.1  else 0.0
        return 1.0 if int(t) % 2 == 0 and (t % 2) < 0.1 else 0.0

    switch_node = nengo.Node(switch)

    # ---- Counter ensemble ----
    counter = nengo.Ensemble(
        n_neurons=500,
        dimensions=1,
        neuron_type=nengo.LIF(),
        radius=30,
    )
    # TODO: use connected counters to count to higher values or just more neurons
    # ---- Integrator (persistent memory) ----
    tau = 0.1
    nengo.Connection(
        counter, counter,
        synapse=tau,
        transform=1.0
    )

    # ---- Increment when switch is on ----
    nengo.Connection(
        switch_node, counter,
        synapse=tau,
        transform=1.0
    )

    # ---- Probes ----
    counter_probe = nengo.Probe(counter, synapse=0.01)
    switch_probe = nengo.Probe(switch_node)
