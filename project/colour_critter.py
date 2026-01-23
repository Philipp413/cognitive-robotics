import grid
import nengo
import nengo.spa as spa
import numpy as np 


#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#########
#  M   R#
#R#R#B#R#
# # # # #
#G Y   R#
#########
"""


#### Preliminaries - this sets up the agent and the environment ################ 
class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=int(4))

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2) 

#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

noise_val = 0.1 # how much noise there will be in the colour info
D = 64

#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()

vocab = spa.Vocabulary(D)
vocab.parse("YES+NO")

vocab2 = spa.Vocabulary(D,max_similarity=0.1)
vocab2.parse("G+R+M+Y+B+W")

col_array = [col_values[c] for c in col_values]

col_vocab = {
    0: "W",
    1: "G",
    2: "R",
    3: "B",
    4: "M",
    5: "Y",
}

with model:
    
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)

    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 15.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        
    movement = nengo.Node(move, size_in=2)
    
    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    def detect(t): # [-0.5,  0. ,  0.5]
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #--------------------------------------------------------------------------#
    def cell2rgb(t):
        
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    current_color = nengo.Node(cell2rgb)
     
    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-white     #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    def look_ahead(t):
        
        done = False
        
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
            
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            
            if cell.cellcolor > 0:
                done = True
        
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    ahead_color = nengo.Node(look_ahead)    
    
    ### Agent functionality - your code adds to this section ###################
    
    model.cc = spa.State(D,vocab=vocab2) # currently seen color by the agend
    
    def signal_to_sp(x):
        
        distances = np.linalg.norm(col_array - x,axis=1)
        
        sel_col = np.argmin(distances)
        
        return vocab2[col_vocab[sel_col]].v
    
    nengo.Connection(current_color, model.cc.input,function=signal_to_sp)
    
    model.seen_red = spa.State(D,vocab=vocab) # if the agend has seen red as the last color
    
    model.cleanup = spa.AssociativeMemory(input_vocab=vocab, wta_output=True)
    nengo.Connection(model.cleanup.output, model.seen_red.input, synapse=0.01)
    nengo.Connection(model.seen_red.output, model.cleanup.input, synapse=0.01)
    
    # seen_red neuron:
    # init state == FALSE
    # get current_color as input
    # if input == RED : set state == YES
    # if input == WHITE : ignore
    # if input == G+B+M+Y : set state == NO
    actions = spa.Actions(
        'dot(cc, R) --> seen_red=3*YES',
        'dot(cc, G) --> seen_red=10*NO',
        'dot(cc, B) --> seen_red=10*NO',
        'dot(cc, M) --> seen_red=10*NO',
        'dot(cc, Y) --> seen_red=10*NO',
        '0.5 --> '
        )
    
    model.bg = spa.BasalGanglia(actions,input_synapse=0.005)
    model.thalamus = spa.Thalamus(model.bg)
    
    model.seen_white = spa.State(D,vocab=vocab)
    
    model.cleanup_sw = spa.AssociativeMemory(input_vocab=vocab, wta_output=True)
    nengo.Connection(model.cleanup_sw.output, model.seen_white.input, synapse=0.01)
    nengo.Connection(model.seen_white.output, model.cleanup_sw.input, synapse=0.01)
    
    actions_sw = spa.Actions(
        'dot(cc, R) --> seen_white=10*NO',
        'dot(cc, G) --> seen_white=10*NO',
        'dot(cc, B) --> seen_white=10*NO',
        'dot(cc, M) --> seen_white=10*NO',
        'dot(cc, Y) --> seen_white=10*NO',
        'dot(cc, W) --> seen_white=10*YES',
        '0.5 --> '
        )
    
    model.bg_sw = spa.BasalGanglia(actions_sw)
    model.thalamus_sw = spa.Thalamus(model.bg_sw)
    
    model.increase_red_counter = spa.State(D,vocab=vocab)
    model.increase_green_counter = spa.State(D,vocab=vocab)
    model.increase_blue_counter = spa.State(D,vocab=vocab)
    model.increase_magenta_counter = spa.State(D,vocab=vocab)
    model.increase_yellow_counter = spa.State(D,vocab=vocab)
    
    actions_ic = spa.Actions( 
        '1/4 * dot(seen_red, YES) + 1/4 * dot(cc, R) + 1/4 * dot(seen_white, YES) --> increase_red_counter=3*YES, increase_green_counter=10*NO, increase_blue_counter=10*NO, increase_magenta_counter=10*NO, increase_yellow_counter=10*NO',
        '0.5 * dot(seen_red, YES) + 0.5 * dot(cc, G) --> increase_green_counter=3*YES, increase_red_counter=10*NO, increase_blue_counter=10*NO, increase_magenta_counter=10*NO, increase_yellow_counter=10*NO',
        '0.5 * dot(seen_red, YES) + 0.5 * dot(cc, B) --> increase_blue_counter=3*YES, increase_red_counter=10*NO, increase_green_counter=10*NO, increase_magenta_counter=10*NO, increase_yellow_counter=10*NO',
        '0.5 * dot(seen_red, YES) + 0.5 * dot(cc, M) --> increase_magenta_counter=3*YES, increase_red_counter=10*NO, increase_green_counter=10*NO, increase_blue_counter=10*NO, increase_yellow_counter=10*NO',
        '0.5 * dot(seen_red, YES) + 0.5 * dot(cc, Y) --> increase_yellow_counter=3*YES, increase_red_counter=10*NO, increase_green_counter=10*NO, increase_blue_counter=10*NO, increase_magenta_counter=10*NO',
        '0.5 --> increase_red_counter=10*NO, increase_green_counter=10*NO, increase_blue_counter=10*NO, increase_magenta_counter=10*NO, increase_yellow_counter=10*NO'
        )
    
    model.bg_ic = spa.BasalGanglia(actions_ic)
    model.thalamus_ic = spa.Thalamus(model.bg_ic)
    
    tau = 0.1
    
    trigger = nengo.Ensemble(
        n_neurons=100,
        dimensions=1
    )
    nengo.Connection(
        model.increase_red_counter.output, # (D,)
        trigger,
        transform=vocab["YES"].v.reshape(1, -1), # does (1,D) x (D,) = (1,). Checks output similarity with YES
        synapse=0.01
    )
    trigger_node = nengo.Node(
        lambda t, x: 1.0 if x > 0.6 else 0.0,
        size_in=1
    )
    nengo.Connection(trigger, trigger_node, synapse=0.01)
    counter = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=10
    )
    nengo.Connection(counter, counter, synapse=tau) # Memory
    nengo.Connection(trigger_node, counter, synapse=tau) # # Trigger increments counter
    
    trigger_g = nengo.Ensemble(
        n_neurons=100,
        dimensions=1
    )
    nengo.Connection(
        model.increase_green_counter.output, # (D,)
        trigger_g,
        transform=vocab["YES"].v.reshape(1, -1), # does (1,D) x (D,) = (1,). Checks output similarity with YES
        synapse=0.01
    )
    trigger_node_g = nengo.Node(
        lambda t, x: 1.0 if x > 0.8 else 0.0,
        size_in=1
    )
    nengo.Connection(trigger_g, trigger_node_g, synapse=0.01)
    counter_g = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=10
    )
    nengo.Connection(counter_g, counter_g, synapse=tau)
    nengo.Connection(trigger_node_g, counter_g, synapse=tau)
    
    trigger_b = nengo.Ensemble(
        n_neurons=100,
        dimensions=1
    )
    nengo.Connection(
        model.increase_blue_counter.output, # (D,)
        trigger_b,
        transform=vocab["YES"].v.reshape(1, -1), # does (1,D) x (D,) = (1,). Checks output similarity with YES
        synapse=0.01
    )
    trigger_node_b = nengo.Node(
        lambda t, x: 1.0 if x > 0.8 else 0.0,
        size_in=1
    )
    nengo.Connection(trigger_b, trigger_node_b, synapse=0.01)
    counter_b = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=10
    )
    nengo.Connection(counter_b, counter_b, synapse=tau)
    nengo.Connection(trigger_node_b, counter_b, synapse=tau)
    
    trigger_m = nengo.Ensemble(
        n_neurons=100,
        dimensions=1
    )
    nengo.Connection(
        model.increase_magenta_counter.output, # (D,)
        trigger_m,
        transform=vocab["YES"].v.reshape(1, -1), # does (1,D) x (D,) = (1,). Checks output similarity with YES
        synapse=0.01
    )
    trigger_node_m = nengo.Node(
        lambda t, x: 1.0 if x > 0.8 else 0.0,
        size_in=1
    )
    nengo.Connection(trigger_m, trigger_node_m, synapse=0.01)
    counter_m = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=10
    )
    nengo.Connection(counter_m, counter_m, synapse=tau)
    nengo.Connection(trigger_node_m, counter_m, synapse=tau)
    
    trigger_y = nengo.Ensemble(
        n_neurons=100,
        dimensions=1
    )
    nengo.Connection(
        model.increase_yellow_counter.output, # (D,)
        trigger_y,
        transform=vocab["YES"].v.reshape(1, -1), # does (1,D) x (D,) = (1,). Checks output similarity with YES
        synapse=0.01
    )
    trigger_node_y = nengo.Node(
        lambda t, x: 1.0 if x > 0.8 else 0.0,
        size_in=1
    )
    nengo.Connection(trigger_y, trigger_node_y, synapse=0.01)
    counter_y = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=10
    )
    nengo.Connection(counter_y, counter_y, synapse=tau)
    nengo.Connection(trigger_node_y, counter_y, synapse=tau)
    
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    # walldist represents the wall distance as vector. 
    walldist = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4) # group of neurons that represent a vector
    nengo.Connection(proximity_sensors, walldist)

    #For now, all our agent does is wall avoidance. It uses values of the radar
    #to: a) turn away from walls on the sides and b) slow down in function of 
    #the distance to the wall ahead, reversing if it is really close
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn
    
    #the movement function is only driven by information from the radar, so we
    #can connect the radar ensemble to the output node with this function 
    #directly. In the assignment, you will need intermediate steps
    nengo.Connection(walldist, movement, function=movement_func)  



    
    
    
 