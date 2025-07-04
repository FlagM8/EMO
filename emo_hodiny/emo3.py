import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from deap import algorithms, base, creator, tools, gp
import operator
import random
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# Define the Santa Fe trail map (1 = food, 0 = empty)
def create_santa_fe_trail():
    trail = np.zeros((32, 32), dtype=int)
    
    # Define the Santa Fe trail coordinates
    trail_coordinates = [
        (0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
        (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13),
        (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20),
        (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26), (12, 27),
        (12, 28), (12, 29), (12, 30), (12, 31), (13, 31), (14, 31), (15, 31),
        (16, 31), (17, 31), (18, 31), (19, 31), (20, 31), (21, 31), (22, 31),
        (23, 31), (24, 31), (24, 30), (24, 29), (24, 28), (24, 27), (24, 26),
        (24, 25), (24, 24), (24, 23), (24, 22), (24, 21), (24, 20), (24, 19),
        (24, 18), (24, 17), (24, 16), (24, 15), (24, 14), (24, 13), (24, 12),
        (24, 11), (24, 10), (24, 9), (24, 8), (24, 7), (24, 6), (24, 5), (24, 4),
        (24, 3), (24, 2), (24, 1), (24, 0)
    ]
    
    # Set the trail with food
    for x, y in trail_coordinates:
        trail[y, x] = 1
    
    return trail, len(trail_coordinates)

# Santa Fe trail
TRAIL, TOTAL_FOOD = create_santa_fe_trail()

# Ant simulator
class AntSimulator:
    NORTH, EAST, SOUTH, WEST = range(4)
    LEFT = {NORTH: WEST, EAST: NORTH, SOUTH: EAST, WEST: SOUTH}
    RIGHT = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}
    AHEAD = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}
    
    def __init__(self, max_moves=600):
        self.moves = 0
        self.eaten = 0
        self.max_moves = max_moves
        self.trail = TRAIL.copy()
        self.position = (0, 0)
        self.direction = self.EAST
        self.eaten_positions = []
        self.positions_history = [(0, 0)]
    
    def sense_food(self):
        x, y = self.position
        dx, dy = self.AHEAD[self.direction]
        new_x, new_y = x + dx, y + dy
        
        # Wrap around the grid
        new_x %= self.trail.shape[1]
        new_y %= self.trail.shape[0]
        
        return self.trail[new_y, new_x] == 1
    
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            x, y = self.position
            dx, dy = self.AHEAD[self.direction]
            new_x, new_y = x + dx, y + dy
            
            # Wrap around the grid
            new_x %= self.trail.shape[1]
            new_y %= self.trail.shape[0]
            
            self.position = (new_x, new_y)
            self.positions_history.append((new_x, new_y))
            
            if self.trail[new_y, new_x] == 1:
                self.trail[new_y, new_x] = 0
                self.eaten += 1
                self.eaten_positions.append((new_x, new_y))
    
    def turn_left(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.direction = self.LEFT[self.direction]
            self.positions_history.append(self.position)
    
    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.direction = self.RIGHT[self.direction]
            self.positions_history.append(self.position)
    
    def run(self, routine):
        self.__init__()
        
        while self.moves < self.max_moves and self.eaten < TOTAL_FOOD:
            routine(self)
        
        return self.eaten

# Function set - these functions need to be defined correctly for DEAP
def if_food_ahead(out1, out2):
    def _if_food_ahead(ant):
        if ant.sense_food():
            out1(ant)
        else:
            out2(ant)
    return _if_food_ahead

def prog2(out1, out2):
    def _prog2(ant):
        out1(ant)
        out2(ant)
    return _prog2

def prog3(out1, out2, out3):
    def _prog3(ant):
        out1(ant)
        out2(ant)
        out3(ant)
    return _prog3

def move_forward(ant):
    ant.move_forward()

def turn_left(ant):
    ant.turn_left()

def turn_right(ant):
    ant.turn_right()

# Set up the DEAP framework
def setup_deap():
    # Clear any existing creators to avoid errors if rerunning
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individual' in creator.__dict__:
        del creator.Individual
        
    # Create fitness and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    pset = gp.PrimitiveSet("ANT", 0)
    pset.addPrimitive(if_food_ahead, 2)
    pset.addPrimitive(prog2, 2)
    pset.addPrimitive(prog3, 3)
    pset.addTerminal(move_forward, name="forward")
    pset.addTerminal(turn_left, name="left")
    pset.addTerminal(turn_right, name="right")
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function
    def evaluate(individual):
        routine = gp.compile(individual, pset)
        ant = AntSimulator()
        return ant.run(routine),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Configure bloat control
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=100))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=100))
    
    return toolbox, pset

# Visualize the decision tree of the best individual
def plot_tree(individual, pset):
    nodes, edges, labels = gp.graph(individual)

    # Create directed graph
    g = nx.DiGraph()
    for i in nodes:
        g.add_node(i)
    for i, j in edges:
        g.add_edge(i, j)

    # Set positions using a tree layout
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(g, pos, node_size=900, node_color="lightblue")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels=labels)
    plt.title("Best Solution Tree")
    plt.axis("off")
    plt.savefig('ant_tree.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize the ant's path
def visualize_ant_path(best_individual, pset):
    # Compile and run the best individual
    routine = gp.compile(best_individual, pset)
    ant = AntSimulator()
    ant.run(routine)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid with food
    grid = TRAIL.copy()
    
    # Create animation
    def update(frame):
        ax.clear()
        
        # Plot the grid
        cmap = mcolors.ListedColormap(['white', 'forestgreen'])
        ax.imshow(grid, cmap=cmap, origin='upper')
        
        # Add grid lines
        for i in range(grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(grid.shape[1] + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # Show eaten food up to this frame
        for i in range(min(frame, len(ant.eaten_positions))):
            x, y = ant.eaten_positions[i]
            ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, fill=True, color='white', alpha=0.7))
        
        # Show the ant's path
        path_length = min(frame + 1, len(ant.positions_history))
        path_x = [pos[0] for pos in ant.positions_history[:path_length]]
        path_y = [pos[1] for pos in ant.positions_history[:path_length]]
        ax.plot(path_x, path_y, 'r-', linewidth=1.5, alpha=0.7)
        
        # Show the ant
        if frame < len(ant.positions_history):
            ant_x, ant_y = ant.positions_history[frame]
            ax.add_patch(Rectangle((ant_x-0.4, ant_y-0.4), 0.8, 0.8, fill=True, color='red'))
        
        # Set axis labels and limits
        ax.set_xticks(np.arange(0, grid.shape[1], 1))
        ax.set_yticks(np.arange(0, grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)
        
        # Add info text
        if frame < len(ant.positions_history):
            food_eaten = sum(1 for i in range(min(frame, len(ant.eaten_positions))))
            ax.set_title(f"Step: {frame}, Food eaten: {food_eaten}/{TOTAL_FOOD}")
        else:
            food_eaten = len(ant.eaten_positions)
            ax.set_title(f"Finished! Food eaten: {food_eaten}/{TOTAL_FOOD}")
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(ant.positions_history)+30, interval=50, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return ani

def main():
    # Setup DEAP
    toolbox, pset = setup_deap()
    
    # Create the population and run the algorithm
    population = toolbox.population(n=300)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the genetic algorithm
    population, logbook = algorithms.eaSimple(population, toolbox, 
                                             cxpb=0.8, mutpb=0.1, 
                                             ngen=50, stats=stats, 
                                             halloffame=hof, verbose=True)
    
    # Display the best individual
    best = hof[0]
    print(f"Best individual fitness: {best.fitness.values[0]}")
    print(f"Food eaten: {best.fitness.values[0]}/{TOTAL_FOOD}")
    print(f"Best individual: {best}")
    
    # Visualize the tree
    plot_tree(best, pset)
    
    # Visualize the ant's path
    ani = visualize_ant_path(best, pset)
    
    return best, pset, ani

if __name__ == "__main__":
    best, pset, ani = main()