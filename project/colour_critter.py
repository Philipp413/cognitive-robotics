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
vocab2 = spa.Vocabulary(D,max_similarity=0.1)

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
        max_speed = 20.0
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

    # random exploration? same as wall avoidance? 
    # 

    # when agent encounters a square, it should be able to correctly recognise it.
    # use current_color. Make basal ganglia actions like in exercise 4.2 
    # Rules: Compute dot(current_color, RED) -> action
    
    # to detect 2 coloured sequences in a row: have some form of memory
    # After seeing one color, there can be white tiles in between before the next. So keep looping color you have seen in memory? 
    # Like evercise 4.2: keep looping last seen color, switch is then second color. And then second color becomes first color and wait for new switch
    # for each colour that it encounters following red, it should count how often that happens
    # Where to store the counters? In memory that decays? so counter resets eventually? 
    # (last_seen=RED x curr_color=RED) and (seen_white) -> redCounter +=1 
    # (last_seen=RED x curr_color=Any different) -> Any different Counter +=1 
    # else: dont change counter
    # Problem: How to make sure that after red there is one white square before red again?
    # seen_white: maybe one ensemble that is just for counting if a white square has been visited, so 1. And goes to 0 if visit color
    
    # TODO: convert 0 to 1 colour input in -1 and 1 range to have large 
    
    vocab2.parse("G+R+M+Y+B+W")
    model.cc = spa.State(D,vocab=vocab2) # represent the last seen color 
    
    def signal_to_spa(xs):
        """ Return spa vector corresponding to input
        
        x - shape (3,)
        
        111 -> W
        010 -> G
        100 -> R
        001 -> B
        101 -> M
        110 -> Y
        """
        x,y,z = xs
        
        if x < 0.5:
            # its G or B
            if y < 0.5:
                return 3*vocab2["B"].v.reshape(D,) 
            return 3*vocab2["G"].v.reshape(D,) 
        # its W, R, M, Y
        if y < 0.5:
            # its R or M
            if z < 0.5:
                return 3*vocab2["R"].v.reshape(D,) 
            return 3*vocab2["M"].v.reshape(D,)    
        # its W or Y
        if z < 0.5:
            return 3*vocab2["Y"].v.reshape(D,)
            
        return 3*vocab2["W"].v.reshape(D,)
    
    nengo.Connection(current_color, model.cc.input,function=signal_to_spa)
    
    # spa State = represent a vector, optionally with memory to remember the vector. 
    vocab.parse("YES+NO")
    model.seen_red = spa.State(D,vocab=vocab) # represent the last seen color 
    
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
    
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)
    
    # seen_white neuron:
    # init state as YES
    # get current_color as input
    # if input == color : set state to NO
    # if input == white: set state to YES
    # use signal delays to time everything
    
    # increase_counter? neuron:
    # (last_seen=RED x curr_color=RED) and (seen_white) -> redCounter +=1 
    # (last_seen=RED x curr_color=Any different) -> Any different Counter +=1 
    
    # todo: find out how the spa stuff can output something 
    
    # how does the flicker wheel output something?
    # use flicker wheel only as counter?
    
    # current color != ahead color: use this to detect color switch? 
    
    # flicker has switch, but node can also act as switch
    # what to make of this? 
    
    # How to build?
    # keep streaming current_color
    # keep last_seen in memory
    # if new different second color or red
    # spike 1 and 
    
    # what about? node last seen == RED. Which broadcasts a 1 if it last seen red and 0 if not, to cancel out other signals

    # use P3 spar 4.3 how to use: when flicker in input, cycle to next letter
    #  
    
    # counter
    # 5 counters for 5 different colours. 
    # lecturer: use SPA or another approach from lecture 6 - does he mean "exploiting multiple time scales"?
    # need to get something to a high value to cause a change in the counter
    # want the counter to go to a high enough value 
    # look at how analog clocks do this
    
    # convolving vector with one element with itself cycles element through
    # use this as counter? 
    # does this decay in memory? 
    # better than using spa? 
    
    # if using SPA: do we need first node, then ensemble and then spa element? 
    # or do you put them directly into spa? 

    # If using look_ahead() function:
    # if current color is red, increase counter of look_ahead by 1
    # And then no need for memory of last seen color? 
    # Problem: How to make sure that after red there is one white square before red again?

    # to compare two spa actions: use spa.Compare
    # you should be able to just include  + cmp on the basal ganglia rule
    # SPA system create a network cmp = spa.Compare(dimensions) and use two cortical rules cmp_A = state1 and cmp_B = state2.
    
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    # all input nodes should feed into the same ensemble? 
    # the weights of walldist are tuned such that? 
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



    
    
    
 