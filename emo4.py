import numpy as np
import pygame
import random
from deap import base, creator, tools, algorithms, gp
import operator
import time
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle, Rectangle
import math

# Original Santa Fe trail
SANTA_FE_TRAIL = [
    (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
    (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
    (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18),
    (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25),
    (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31), (6, 31),
    (7, 31), (8, 31), (9, 31), (10, 31), (11, 31), (12, 31), (13, 31),
    (14, 31), (15, 31), (16, 31), (17, 31), (18, 31), (19, 31), (20, 31),
    (21, 31), (22, 31), (23, 31), (24, 31), (25, 31), (26, 31), (27, 31),
    (28, 31), (29, 31), (30, 31), (31, 31), (31, 30), (31, 29), (31, 28),
    (31, 27), (31, 26), (31, 25), (31, 24), (31, 23), (31, 22), (31, 21),
    (31, 20), (31, 19), (31, 18), (31, 17), (31, 16), (31, 15), (31, 14),
    (31, 13), (31, 12), (31, 11), (31, 10), (31, 9), (31, 8), (31, 7),
    (31, 7), (31, 8), (31, 9), (31, 10), (31, 11), (31, 12), (31, 13),
    (31, 14)
]

def generate_random_trail(num_food=90, grid_size=32, coherence=0.7):
    """
    Generate a random trail with some coherence (adjacent food positions)
    
    Args:
        num_food: Number of food pieces to place
        grid_size: Size of the grid (grid_size x grid_size)
        coherence: Probability of placing food adjacent to existing food
    
    Returns:
        List of (x, y) food positions
    """
    trail = []
    occupied = set()
    
    # Place the first food randomly
    x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    trail.append((x, y))
    occupied.add((x, y))
    
    # Define possible moves (4-connected neighborhood)
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Place remaining food
    for _ in range(num_food - 1):
        # Decide whether to place adjacent to existing food
        if random.random() < coherence and trail:
            # Choose a random existing food piece
            base_x, base_y = random.choice(trail)
            
            # Try to find an unoccupied adjacent position
            adjacent_positions = []
            for dx, dy in moves:
                new_x, new_y = (base_x + dx) % grid_size, (base_y + dy) % grid_size
                if (new_x, new_y) not in occupied:
                    adjacent_positions.append((new_x, new_y))
            
            # If found, place food there
            if adjacent_positions:
                x, y = random.choice(adjacent_positions)
            else:
                # If no adjacent positions available, place randomly
                while True:
                    x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
                    if (x, y) not in occupied:
                        break
        else:
            # Place randomly
            while True:
                x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
                if (x, y) not in occupied:
                    break
        
        trail.append((x, y))
        occupied.add((x, y))
    
    return trail

class AntSimulator:
    def __init__(self, trail):
        self.trail = trail
        self.reset()
        
    def reset(self):
        self.x = 0
        self.y = 0
        self.direction = 0  # 0: east, 1: south, 2: west, 3: north
        self.eaten = 0
        self.moves = 0
        self.trail_grid = np.zeros((32, 32), dtype=bool)
        for x, y in self.trail:
            self.trail_grid[x, y] = True
        self.visited = set()
        self.movement_history = [(0, 0)]  # Track all positions the ant visits
    
    def turn_left(self):
        self.direction = (self.direction - 1) % 4
        self.moves += 1
        return self.moves < 600
    
    def turn_right(self):
        self.direction = (self.direction + 1) % 4
        self.moves += 1
        return self.moves < 600
    
    def move_forward(self):
        if self.direction == 0:  # east
            self.x = (self.x + 1) % 32
        elif self.direction == 1:  # south
            self.y = (self.y + 1) % 32
        elif self.direction == 2:  # west
            self.x = (self.x - 1) % 32
        else:  # north
            self.y = (self.y - 1) % 32
            
        self.moves += 1
        # Add current position to movement history
        self.movement_history.append((self.x, self.y))
        
        if self.trail_grid[self.x, self.y] and (self.x, self.y) not in self.visited:
            self.eaten += 1
            self.visited.add((self.x, self.y))
        return self.moves < 600
    
    def sense_food(self):
        ahead_x = self.x
        ahead_y = self.y
        
        if self.direction == 0:  # east
            ahead_x = (self.x + 1) % 32
        elif self.direction == 1:  # south
            ahead_y = (self.y + 1) % 32
        elif self.direction == 2:  # west
            ahead_x = (self.x - 1) % 32
        else:  # north
            ahead_y = (self.y - 1) % 32
            
        return self.trail_grid[ahead_x, ahead_y]

def if_food_ahead(out1, out2):
    def _if_food_ahead():
        if ant.sense_food():
            out1()
        else:
            out2()
    return _if_food_ahead

def prog2(out1, out2):
    def _prog2():
        out1()
        out2()
    return _prog2

def prog3(out1, out2, out3):
    def _prog3():
        out1()
        out2()
        out3()
    return _prog3

def move_forward():
    ant.move_forward()

def turn_left():
    ant.turn_left()

def turn_right():
    ant.turn_right()

# Pygame visualization setup
pygame.init()
CELL_SIZE = 20
WINDOW_SIZE = (32 * CELL_SIZE + 200, 32 * CELL_SIZE)  # Added extra space for info panel
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Santa Fe Artificial Ant")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Initialize font
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)

def draw_grid():
    screen.fill(WHITE)
    
    # Draw trail and visited cells
    for x in range(32):
        for y in range(32):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if ant.trail_grid[x, y]:
                if (x, y) in ant.visited:
                    pygame.draw.rect(screen, BLUE, rect)
                else:
                    pygame.draw.rect(screen, GREEN, rect)
    
    # Draw movement history trail (past positions)
    for i, (x, y) in enumerate(ant.movement_history[:-1]):  # Skip current position
        # Calculate color based on how old the position is (fading from orange to light gray)
        age_ratio = i / max(1, len(ant.movement_history) - 1)
        if len(ant.movement_history) > 10000:
            # Only show the last 100 positions with color
            if i < len(ant.movement_history) - 100:
                continue
            age_ratio = (i - (len(ant.movement_history) - 100)) / 100
            
        color = tuple(int(a + (b - a) * age_ratio) for a, b in zip(ORANGE, LIGHT_GRAY))
        
        # Draw a small circle at this position
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        size = max(2, int(CELL_SIZE // 6))
        pygame.draw.circle(screen, color, (center_x, center_y), size)
    
    # Connect points with lines to show movement path
    if len(ant.movement_history) > 1:
        points = [(x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2) 
                 for x, y in ant.movement_history[-min(100, len(ant.movement_history)):]]
        pygame.draw.lines(screen, PURPLE, False, points, 2)
        
    # Draw ant with direction indicator
    ant_rect = pygame.Rect(ant.x * CELL_SIZE, ant.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, ant_rect)
    
    # Draw direction indicator
    center_x = ant.x * CELL_SIZE + CELL_SIZE // 2
    center_y = ant.y * CELL_SIZE + CELL_SIZE // 2
    if ant.direction == 0:  # east
        pygame.draw.line(screen, BLACK, (center_x, center_y), (center_x + CELL_SIZE//2, center_y), 3)
    elif ant.direction == 1:  # south
        pygame.draw.line(screen, BLACK, (center_x, center_y), (center_x, center_y + CELL_SIZE//2), 3)
    elif ant.direction == 2:  # west
        pygame.draw.line(screen, BLACK, (center_x, center_y), (center_x - CELL_SIZE//2, center_y), 3)
    else:  # north
        pygame.draw.line(screen, BLACK, (center_x, center_y), (center_x, center_y - CELL_SIZE//2), 3)
    
    # Draw info panel
    panel_rect = pygame.Rect(32 * CELL_SIZE, 0, 200, 32 * CELL_SIZE)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect)
    
    # Draw statistics
    info_texts = [
        f"Food eaten: {ant.eaten}/{len(TRAIL)}",
        f"Moves: {ant.moves}/600",
        f"Remaining: {len(TRAIL) - ant.eaten}",
        "",
        "Controls:",
        "Space: Pause/Resume",
        "+/-: Speed up/down",
        "ESC: Quit"
    ]
    
    for i, text in enumerate(info_texts):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (32 * CELL_SIZE + 10, 20 + i * 25))
    
    pygame.display.flip()

def plot_tree_custom(expr, output_file='tree.png'):
    """Plot the tree expression using a custom algorithm without NetworkX"""
    nodes, edges, labels = gp.graph(expr)
    
    # Create figure and axis - much larger figure for dense trees
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Layout algorithm parameters - increased spacing
    root = 0
    node_radius = 0.6  # Bigger nodes
    level_height = 2.5  # More vertical space
    
    # Calculate the positions using a simple tree layout algorithm
    def get_subtree_width(node, depth=0):
        """Calculate the width needed for a subtree"""
        children = [j for i, j in edges if i == node]
        if not children:
            return 2  # Leaf nodes have more width now
        
        # Allocate more space for deeper trees
        width = sum(get_subtree_width(child, depth + 1) for child in children)
        # Add extra space for better separation between siblings
        return width + (len(children) - 1) * 0.5  # Add spacing between siblings
    
    # Calculate node positions
    positions = {}
    
    def position_subtree(node, x, y, width):
        """Position a subtree recursively"""
        positions[node] = (x, -y)  # Store position (negate y for top-down tree)
        
        children = [j for i, j in edges if i == node]
        if not children:
            return
        
        # Calculate positions for children
        child_widths = [get_subtree_width(child) for child in children]
        total_width = sum(child_widths) + (len(children) - 1) * 0.8  # Add spacing
        
        # Scale child widths to fit the allocated width
        if total_width > 0:
            # Don't scale widths, use fixed spacing
            scaled_widths = child_widths
        else:
            scaled_widths = child_widths
        
        # Position each child
        current_x = x - total_width/2
        for i, child in enumerate(children):
            child_width = scaled_widths[i]
            child_x = current_x + child_width/2
            position_subtree(child, child_x, y + level_height, child_width)
            current_x += child_width + 0.8  # Add fixed spacing between siblings
    
    # Start the layout from the root
    total_width = get_subtree_width(root)
    position_subtree(root, 0, 0, total_width)
    
    # Draw the edges
    for i, j in edges:
        start = positions[i]
        end = positions[j]
        
        # Draw line
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=1.5, zorder=1)
        
        # Draw arrow
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Normalize direction
            dx, dy = dx/length, dy/length
            # Calculate arrow position (near the end point)
            arrow_size = 0.1
            arrow_x = end[0] - dx * node_radius
            arrow_y = end[1] - dy * node_radius
            ax.arrow(arrow_x, arrow_y, dx * arrow_size, dy * arrow_size, 
                    head_width=0.15, head_length=0.2, fc='k', ec='k', zorder=2)
    
    # Draw the nodes with colors based on function
    for i, label in labels.items():
        x, y = positions[i]
        
        # Color nodes based on function
        if label == 'if_food_ahead':
            color = 'lightblue'
        elif label in ('prog2', 'prog3'):
            color = 'lightgreen'
        elif label == 'move_forward':
            color = 'salmon'
        elif label == 'turn_left':
            color = 'yellow'
        elif label == 'turn_right':
            color = 'orange'
        else:
            color = 'white'
        
        # Create circle and text
        circle = Circle((x, y), node_radius, fill=True, color=color, ec='black', zorder=3)
        ax.add_patch(circle)
        
        # Add label with wrapped text - larger font
        wrapped_label = label
        if len(label) > 12:
            wrapped_label = label[:10] + "..."
        ax.text(x, y, wrapped_label, ha='center', va='center', fontsize=10, weight='bold', zorder=4)
    
    # Set equal aspect and disable axis
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    # Set limits with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding = 2.0  # More padding
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    # Add a legend with larger elements
    legend_entries = [
        Rectangle((0, 0), 1, 1, fc='lightblue', ec='black', label='if_food_ahead'),
        Rectangle((0, 0), 1, 1, fc='lightgreen', ec='black', label='prog2/prog3'),
        Rectangle((0, 0), 1, 1, fc='salmon', ec='black', label='move_forward'),
        Rectangle((0, 0), 1, 1, fc='yellow', ec='black', label='turn_left'),
        Rectangle((0, 0), 1, 1, fc='orange', ec='black', label='turn_right')
    ]
    ax.legend(handles=legend_entries, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=5, fancybox=True, shadow=True, fontsize=12)
    
    # Adjust layout and save with higher DPI
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def show_tree_on_pygame(tree_image_path):
    """Display the tree image on a separate Pygame window with pan and zoom"""
    # Load the tree image
    tree_image = pygame.image.load(tree_image_path)
    original_image = tree_image.copy()  # Keep a copy of the original image
    
    # Get image dimensions
    img_width, img_height = tree_image.get_width(), tree_image.get_height()
    
    # Create a window that fits on screen
    screen_info = pygame.display.Info()
    window_width = min(img_width, screen_info.current_w - 100)
    window_height = min(img_height, screen_info.current_h - 100)
    
    tree_window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Evolved Decision Tree - Use Mouse Wheel to Zoom, Drag to Pan")
    
    # Initialize variables for panning and zooming
    offset_x, offset_y = 0, 0
    zoom_level = 1.0
    dragging = False
    drag_start = (0, 0)
    
    # Font for instructions
    font = pygame.font.SysFont('Arial', 16)
    
    # Function to redraw the image with current pan and zoom
    def redraw():
        tree_window.fill((240, 240, 240))  # Light gray background
        
        # Calculate zoom and offset
        current_width = int(img_width * zoom_level)
        current_height = int(img_height * zoom_level)
        
        if current_width > 0 and current_height > 0:  # Prevent zero size
            # Create zoomed image
            zoomed_image = pygame.transform.smoothscale(original_image, (current_width, current_height))
            
            # Calculate display position with offset
            display_x = max(0, min(offset_x, window_width - current_width))
            display_y = max(0, min(offset_y, window_height - current_height))
            
            if current_width < window_width:
                display_x = (window_width - current_width) // 2
            if current_height < window_height:
                display_y = (window_height - current_height) // 2
            
            # Blit the image
            tree_window.blit(zoomed_image, (display_x, display_y))
        
        # Draw instructions
        instructions = [
            "Mouse wheel: Zoom in/out",
            "Drag: Pan the view",
            "Space: Reset view",
            "ESC: Close window"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, (0, 0, 0))
            tree_window.blit(text_surface, (10, 10 + i * 20))
        
        pygame.display.flip()
    
    # Wait for user to close the window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False
                elif event.key == pygame.K_SPACE:
                    # Reset view
                    offset_x = (window_width - img_width) // 2
                    offset_y = (window_height - img_height) // 2
                    zoom_level = 1.0
                    redraw()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left button
                    dragging = True
                    drag_start = event.pos
                elif event.button == 4:  # Scroll up
                    zoom_level = min(5.0, zoom_level * 1.1)
                    redraw()
                elif event.button == 5:  # Scroll down
                    zoom_level = max(0.1, zoom_level / 1.1)
                    redraw()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                # Update offset based on mouse movement
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                offset_x += dx
                offset_y += dy
                drag_start = event.pos
                redraw()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                window_width, window_height = event.size
                tree_window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                redraw()
        
        # Initial draw
        if waiting:
            redraw()
            waiting = True  # Just to ensure the loop continues
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Santa Fe Artificial Ant Problem')
    parser.add_argument('--random', action='store_true', help='Use random food trail instead of Santa Fe')
    parser.add_argument('--food', type=int, default=90, help='Number of food pieces in random trail')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--coherence', type=float, default=0.7, help='Coherence of random trail (0-1)')
    parser.add_argument('--gens', type=int, default=40, help='Number of generations')
    parser.add_argument('--pop', type=int, default=300, help='Population size')
    parser.add_argument('--cxpb', type=float, default=0.8, help='Crossover probability')
    parser.add_argument('--mutpb', type=float, default=0.2, help='Mutation probability')
    parser.add_argument('--tournament', type=int, default=7, help='Tournament size')
    parser.add_argument('--elite', type=int, default=10, help='Number of elite individuals to preserve')
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(int(time.time()))
    
    # Select trail
    if args.random:
        TRAIL = generate_random_trail(num_food=args.food, coherence=args.coherence)
        print(f"Generated random trail with {len(TRAIL)} food pieces")
    else:
        TRAIL = SANTA_FE_TRAIL
        print(f"Using original Santa Fe trail with {len(TRAIL)} food pieces")
    
    ant = AntSimulator(TRAIL)
    
    # Initialize DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(prog2, 2)
    pset.addPrimitive(prog3, 3)
    pset.addPrimitive(if_food_ahead, 2)
    pset.addTerminal(move_forward)
    pset.addTerminal(turn_left)
    pset.addTerminal(turn_right)

    # Evaluation function with improved heuristics
    def evalArtificialAnt(individual):
        routine = gp.compile(individual, pset)
        ant.reset()
        
        # Keep track of time spent without eating
        last_eaten = 0
        idle_penalty = 0
        
        while ant.moves < 600:
            prev_eaten = ant.eaten
            routine()
            
            # If we ate something, reset the idle counter
            if ant.eaten > prev_eaten:
                last_eaten = ant.moves
                idle_penalty = 0
            else:
                # Penalize too much time without eating
                idle_penalty = max(0, (ant.moves - last_eaten) / 100)
        
        # Base fitness is the number of food eaten
        fitness = ant.eaten
        
        # Penalize solutions that spend too much time without finding food
        fitness = fitness - idle_penalty
        
        # Penalize overly complex solutions (parsimony pressure)
        tree_size = len(individual)
        parsimony_penalty = 0.001 * max(0, tree_size - 50)  # Only penalize if bigger than 50 nodes
        
        return fitness - parsimony_penalty,

    # Genetic Algorithm parameters
    toolbox = base.Toolbox()
    
    # Tree generation with variable depth to create more diverse initial population
    toolbox.register("expr_full", gp.genFull, pset=pset, min_=2, max_=6)
    toolbox.register("expr_grow", gp.genGrow, pset=pset, min_=2, max_=6)
    
    def generate_diverse():
        # Choose between full and grow method
        method = random.choice([toolbox.expr_full, toolbox.expr_grow])
        return method()
    
    toolbox.register("expr", generate_diverse)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalArtificialAnt)
    toolbox.register("select", tools.selTournament, tournsize=args.tournament)
    toolbox.register("mate", gp.cxOnePoint)
    
    # More sophisticated mutation
    toolbox.register("expr_mut_full", gp.genFull, min_=0, max_=2)
    toolbox.register("expr_mut_grow", gp.genGrow, min_=0, max_=2)
    
    def mutate_diverse(individual, expr):
        # 20% chance of shrink mutation, 80% chance of subtree mutation
        if random.random() < 0.2:
            return gp.mutShrink(individual)
        else:
            return gp.mutUniform(individual, expr, pset=pset)
    
    toolbox.register("mutate", mutate_diverse, expr=toolbox.expr_mut_grow)
    
    # Setup statistics
    pop = toolbox.population(n=args.pop)
    hof = tools.HallOfFame(args.elite)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Custom evolutionary algorithm with elitism
    def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                           halloffame=None, verbose=__debug__):
        """This algorithm is similar to DEAP eaSimple but implements elitism directly"""
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - len(halloffame))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Add the elite individuals to the offspring
            offspring.extend(halloffame)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            # Check for convergence
            if halloffame[0].fitness.values[0] >= len(TRAIL):
                print(f"Found perfect solution at generation {gen}")
                break

        return population, logbook
    
    # Run the evolutionary algorithm
    pop, log = eaSimpleWithElitism(pop, toolbox, args.cxpb, args.mutpb, args.gens, 
                               stats=stats, halloffame=hof, verbose=True)
    
    print(f"\nBest individual fitness: {hof[0].fitness.values[0]}")
    print(f"Total food on trail: {len(TRAIL)}")
    
    # Show best individual structure
    best_tree = str(hof[0])
    print(f"\nBest individual size: {len(hof[0])}")
    print(f"Best individual structure: {best_tree[:50]}..." if len(best_tree) > 50 else best_tree)
    
    # Visualize best solution
    best_routine = gp.compile(hof[0], pset)
    ant.reset()
    running = True
    paused = False
    clock = pygame.time.Clock()
    delay = 200  # Default increased to 200ms (slower)
    
    last_step_time = pygame.time.get_ticks()
    show_tree = True  # Flag to show tree after simulation
    
    while running:
        # Limit to 60 FPS
        clock.tick(60)
        
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                show_tree = False  # Don't show tree if user quits
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    delay = max(10, delay - 10)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    delay = min(500, delay + 10)
                elif event.key == pygame.K_t:
                    # Show tree on demand
                    tree_file = plot_tree_custom(hof[0])
                    pygame.display.iconify()  # Minimize main window
                    show_tree_on_pygame(tree_file)
                    pygame.display.set_mode(WINDOW_SIZE)  # Restore main window
        
        # Update ant if not paused and time to update
        if not paused and current_time - last_step_time > delay and ant.moves < 600:
            best_routine()
            last_step_time = current_time
        
        # Always draw even when paused
        draw_grid()
        
        # If simulation completed
        if ant.moves >= 600:
            text = font.render("Simulation completed! Press ESC to quit or T to view the decision tree", True, WHITE)
            text_rect = text.get_rect(center=(32 * CELL_SIZE // 2, 32 * CELL_SIZE - 30))
            screen.blit(text, text_rect)
            pygame.display.flip()
        
    # Show the tree after simulation ends (if not quit by user)
    if show_tree:
        # Generate and save the tree visualization
        tree_file = plot_tree_custom(hof[0])
        show_tree_on_pygame(tree_file)
    
    pygame.quit()