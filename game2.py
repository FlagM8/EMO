import pygame
import pymunk
import pymunk.pygame_util
import neat
import os
import math
import random

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
PHYSICS_STEPS = 1 / 60.0
HUMANOID_COUNT = 50

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Humanoid Walker - Improved")
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
font = pygame.font.SysFont("Arial", 20)

best_fitness_so_far = 0.0

class Humanoid:
    def __init__(self, space, position, collision_type_offset):
        self.space = space
        self.bodies = {}
        self.shapes = {}
        self.joints = {}
        self.initial_pos = position
        self.collision_type_base = collision_type_offset * 10

        self.create_body(position)
        self.is_alive = True
        self.fitness = 0.0
        self.max_x = position[0] 
        self.prev_x = position[0]
        self.prev_velocity = 0.0
        self.steps_taken = 0
        self.ground_contacts = {"left_foot": False, "right_foot": False}
        self.step_time = 0

    def create_body(self, pos):
        x, y = pos
        group_id = 1 + self.collision_type_base

        self.bodies['torso'] = pymunk.Body(15, 150)
        self.bodies['torso'].position = (x, y - 50)
        shape = pymunk.Poly.create_box(self.bodies['torso'], (20, 80))
        shape.filter = pymunk.ShapeFilter(group=group_id, categories=1 << (self.collision_type_base % 32), mask=1)
        shape.color = (0, 0, 255, 255)
        shape.collision_type = self.collision_type_base + 1
        shape.friction = 0.3
        self.shapes['torso'] = shape

        self.bodies['head'] = pymunk.Body(5, 50)
        self.bodies['head'].position = (x, y - 110)
        shape = pymunk.Circle(self.bodies['head'], 20)
        shape.filter = pymunk.ShapeFilter(group=group_id, categories=1 << (self.collision_type_base % 32), mask=1)
        shape.color = (255, 0, 0, 255)
        shape.collision_type = self.collision_type_base + 2
        shape.friction = 0.3
        self.shapes['head'] = shape

        for i in range(2):
            side = "left" if i == 0 else "right"
            x_offset = -15 if i == 0 else 15
            
            # Upper Leg
            self.bodies[f'upper_leg_{i}'] = pymunk.Body(6, 100)
            self.bodies[f'upper_leg_{i}'].position = (x + x_offset, y - 5)
            shape = pymunk.Poly.create_box(self.bodies[f'upper_leg_{i}'], (12, 50))
            shape.filter = pymunk.ShapeFilter(group=group_id, categories=1 << (self.collision_type_base % 32), mask=1)
            shape.color = (0, 255, 0, 255)
            shape.collision_type = self.collision_type_base + 3 + i*3
            shape.friction = 0.3
            self.shapes[f'upper_leg_{i}'] = shape

            # Lower Leg
            self.bodies[f'lower_leg_{i}'] = pymunk.Body(4, 80)
            self.bodies[f'lower_leg_{i}'].position = (x + x_offset, y + 45)
            shape = pymunk.Poly.create_box(self.bodies[f'lower_leg_{i}'], (10, 40))
            shape.filter = pymunk.ShapeFilter(group=group_id, categories=1 << (self.collision_type_base % 32), mask=1)
            shape.color = (0, 200, 50, 255)
            shape.collision_type = self.collision_type_base + 4 + i*3
            shape.friction = 0.3
            self.shapes[f'lower_leg_{i}'] = shape
            
            # Foot
            self.bodies[f'foot_{i}'] = pymunk.Body(2, 20)
            self.bodies[f'foot_{i}'].position = (x + x_offset, y + 75)
            shape = pymunk.Poly.create_box(self.bodies[f'foot_{i}'], (25, 8))
            shape.filter = pymunk.ShapeFilter(group=group_id, categories=1 << (self.collision_type_base % 32), mask=1)
            shape.color = (100, 100, 100, 255)
            shape.collision_type = self.collision_type_base + 5 + i*3
            shape.friction = 1.0 
            self.shapes[f'foot_{i}'] = shape

        for body in self.bodies.values():
            self.space.add(body)
        for shape in self.shapes.values():
            self.space.add(shape)

        # Joints and motors and magic stuff :)
        self.joints['head_joint'] = pymunk.PivotJoint(self.bodies['torso'], self.bodies['head'], (0, -40), (0, 20))
        self.joints['head_joint'].collide_bodies = False
        self.space.add(self.joints['head_joint'])

        for i in range(2):
            x_offset = -10 if i == 0 else 10
            
            # hip joint - now with rotation limits :D
            self.joints[f'hip_joint_{i}'] = pymunk.PivotJoint(
                self.bodies['torso'], self.bodies[f'upper_leg_{i}'], 
                (x_offset, 40), (0, -25)
            )
            self.joints[f'hip_joint_{i}'].collide_bodies = False
            
            # hip moto
            self.joints[f'hip_motor_{i}'] = pymunk.SimpleMotor(self.bodies['torso'], self.bodies[f'upper_leg_{i}'], 0)
            self.joints[f'hip_motor_{i}'].max_force = 800000
            
            # hip rotation limit
            self.joints[f'hip_limit_{i}'] = pymunk.RotaryLimitJoint(
                self.bodies['torso'], self.bodies[f'upper_leg_{i}'], 
                -math.pi/2, math.pi/3
            )
            
            self.space.add(self.joints[f'hip_joint_{i}'], self.joints[f'hip_motor_{i}'], self.joints[f'hip_limit_{i}'])
            
            # knee joint
            self.joints[f'knee_joint_{i}'] = pymunk.PivotJoint(
                self.bodies[f'upper_leg_{i}'], self.bodies[f'lower_leg_{i}'], 
                (0, 25), (0, -20)
            )
            self.joints[f'knee_joint_{i}'].collide_bodies = False
            
            # nee motor
            self.joints[f'knee_motor_{i}'] = pymunk.SimpleMotor(self.bodies[f'upper_leg_{i}'], self.bodies[f'lower_leg_{i}'], 0)
            self.joints[f'knee_motor_{i}'].max_force = 600000
            
            # nee rotation limit (knees don't bend backwards now :D)
            self.joints[f'knee_limit_{i}'] = pymunk.RotaryLimitJoint(
                self.bodies[f'upper_leg_{i}'], self.bodies[f'lower_leg_{i}'], 
                -math.pi/8, math.pi/2
            )
            
            self.space.add(self.joints[f'knee_joint_{i}'], self.joints[f'knee_motor_{i}'], self.joints[f'knee_limit_{i}'])
            
            # ankle joint
            self.joints[f'ankle_joint_{i}'] = pymunk.PivotJoint(
                self.bodies[f'lower_leg_{i}'], self.bodies[f'foot_{i}'], 
                (0, 20), (0, -4)
            )
            self.joints[f'ankle_joint_{i}'].collide_bodies = False
            
            # ankle motor
            self.joints[f'ankle_motor_{i}'] = pymunk.SimpleMotor(self.bodies[f'lower_leg_{i}'], self.bodies[f'foot_{i}'], 0)
            self.joints[f'ankle_motor_{i}'].max_force = 400000
            
            #ankle rotation limit
            self.joints[f'ankle_limit_{i}'] = pymunk.RotaryLimitJoint(
                self.bodies[f'lower_leg_{i}'], self.bodies[f'foot_{i}'], 
                -math.pi/4, math.pi/4
            )
            
            self.space.add(self.joints[f'ankle_joint_{i}'], self.joints[f'ankle_motor_{i}'], self.joints[f'ankle_limit_{i}'])

    def get_inputs(self, wall_x):
        inputs = []
        
        # body orientation and angular velocity 
        inputs.append(self.bodies['torso'].angle / math.pi)
        inputs.append(self.bodies['torso'].angular_velocity / 10.0)
        
        # torso velocity (important for movement allegedly :D)
        inputs.append(self.bodies['torso'].velocity.x / 100.0)
        inputs.append(self.bodies['torso'].velocity.y / 100.0)
        
        #jJoint angles for both legs 
        for i in range(2):
            # Hip angle
            hip_angle = self.bodies[f'upper_leg_{i}'].angle - self.bodies['torso'].angle
            inputs.append(hip_angle / math.pi)
            
            # Knee angle
            knee_angle = self.bodies[f'lower_leg_{i}'].angle - self.bodies[f'upper_leg_{i}'].angle
            inputs.append(knee_angle / math.pi)
            
            # ankle angle
            ankle_angle = self.bodies[f'foot_{i}'].angle - self.bodies[f'lower_leg_{i}'].angle
            inputs.append(ankle_angle / math.pi)
            
            # foot ground contact 
            foot_y = self.bodies[f'foot_{i}'].position.y
            ground_level = SCREEN_HEIGHT - 100
            inputs.append(1.0 if foot_y >= ground_level - 15 else 0.0)

        # head height relative to start
        head_height = (self.bodies['head'].position.y - self.initial_pos[1]) / SCREEN_HEIGHT
        inputs.append(head_height)
        
        # distance traveled
        distance = (self.bodies['torso'].position.x - self.initial_pos[0]) / SCREEN_WIDTH
        inputs.append(distance)

        return inputs

    def apply_outputs(self, outputs):
        if not self.is_alive:
            return

        motor_speed = 4.0 
        
        if len(outputs) >= 6:
            self.joints['hip_motor_0'].rate = outputs[0] * motor_speed
            self.joints['knee_motor_0'].rate = outputs[1] * motor_speed
            self.joints['ankle_motor_0'].rate = outputs[2] * motor_speed
            self.joints['hip_motor_1'].rate = outputs[3] * motor_speed
            self.joints['knee_motor_1'].rate = outputs[4] * motor_speed
            self.joints['ankle_motor_1'].rate = outputs[5] * motor_speed

    def check_fall(self, wall_x=None):
        ground_level = SCREEN_HEIGHT - 100
        margin = -15
        
        # head or torso hit ground detectioin
        if (self.bodies['head'].position.y >= ground_level + margin or 
            self.bodies['torso'].position.y >= ground_level + margin):
            self.is_alive = False
            return
            
        
        # titlt detection
        if abs(self.bodies['torso'].angle) > math.pi/2:
            self.is_alive = False
            return
        
        # wall detection
        if wall_x is not None:
            if self.bodies['torso'].position.x < wall_x - 15:
                self.is_alive = False
                return

    def calculate_fitness(self):
        # distance
        x_distance = self.bodies['torso'].position.x - self.initial_pos[0]
        self.max_x = max(self.max_x, self.bodies['torso'].position.x)
        
        # velocity
        velocity = self.bodies['torso'].velocity.x
        velocity_bonus = max(0, velocity / 50.0)  # Bonus for moving forward
        
        # upright
        torso_angle = abs(self.bodies['torso'].angle)
        upright_bonus = max(0, (math.pi/3 - torso_angle) / (math.pi/3))
        
        # head height - 
        head_height = self.bodies['head'].position.y
        target_height = self.initial_pos[1] - 110
        height_bonus = max(0, 1.0 - abs(head_height - target_height) / 100.0)
        
        # stability bonus (penalize excessive rotation)
        angular_vel = abs(self.bodies['torso'].angular_velocity)
        stability_bonus = max(0, 1.0 - angular_vel / 10.0)
        
        # leg movement bonus (encourage leg usage)
        leg_movement = 0
        for i in range(2):
            leg_movement += abs(self.joints[f'hip_motor_{i}'].rate)
            leg_movement += abs(self.joints[f'knee_motor_{i}'].rate)
            if f'ankle_motor_{i}' in self.joints:
                leg_movement += abs(self.joints[f'ankle_motor_{i}'].rate)
        leg_movement_bonus = min(1.0, leg_movement / 10.0)
        
        # penalize legs too far apart
        leg_distance = abs(self.bodies['upper_leg_0'].position.x - self.bodies['upper_leg_1'].position.x)
        leg_spread_penalty = 0.0
        leg_spread_threshold = 30.0
        if leg_distance > leg_spread_threshold:
            leg_spread_penalty = (leg_distance - leg_spread_threshold) * 50.0 

        # feet facing ground bonus 
        feet_facing_bonus = 0
        for i in range(2):
            ankle_angle = self.bodies[f'foot_{i}'].angle - self.bodies[f'lower_leg_{i}'].angle
            feet_facing_bonus += max(0, 1.0 - abs(ankle_angle) / (math.pi / 2))
        feet_facing_bonus = (feet_facing_bonus / 2.0) * 10.0 

        # penalize torso and head that are too close to the ground
        ground_level = SCREEN_HEIGHT - 100
        torso_y = self.bodies['torso'].position.y
        head_y = self.bodies['head'].position.y
        torso_ground_penalty = max(0, 1.0 - (ground_level - torso_y) / 120.0) * 35.0
        head_ground_penalty = max(0, 1.0 - (ground_level - head_y) / 120.0) * 45.0

        # combine alll
        self.fitness = (
            x_distance * 0.1 +      
            velocity_bonus * 2.0 +             
            upright_bonus * 25.0 +             
            height_bonus * 39.0 +              
            stability_bonus * 10.0 +        
            leg_movement_bonus * 8.0 +  
            feet_facing_bonus             
            - torso_ground_penalty           
            - head_ground_penalty            
            - leg_spread_penalty   
        )
        

        self.step_time += 1
        self.fitness += self.step_time * 0.01

    def remove_from_space(self):
        for joint in self.joints.values():
            if joint in self.space.constraints:
                self.space.remove(joint)
        for shape in self.shapes.values():
            if shape in self.space.shapes:
                self.space.remove(shape)
        for body in self.bodies.values():
            if body in self.space.bodies:
                self.space.remove(body)

def draw_neural_network(surface, genome, config, position, width, height):

    x, y = position
    node_radius = 8
    
    input_names = ['TorsoAng', 'TorsoAngVel', 'VelX', 'VelY', 
                   'Hip0', 'Knee0', 'Ankle0', 'Foot0', 
                   'Hip1', 'Knee1', 'Ankle1', 'Foot1', 
                   'HeadY', 'Distance']
    output_names = ['Hip0', 'Knee0', 'Ankle0', 'Hip1', 'Knee1', 'Ankle1']

    node_positions = {}
    
    genome_config = config.genome_config
    input_keys = genome_config.input_keys
    output_keys = genome_config.output_keys
    
    # input nodes
    for i, key in enumerate(input_keys):
        node_positions[key] = (x, y + (i + 1) * height / (len(input_keys) + 1))
    
    # output nodes
    for i, key in enumerate(output_keys):
        node_positions[key] = (x + width, y + (i + 1) * height / (len(output_keys) + 1))
        
    # hidden nodes
    hidden_nodes_keys = [key for key in genome.nodes.keys() if key not in input_keys and key not in output_keys]
    if hidden_nodes_keys:
        for i, key in enumerate(hidden_nodes_keys):
            node_positions[key] = (x + width / 2, y + (i + 1) * height / (len(hidden_nodes_keys) + 1))

    # draw connections
    for conn in genome.connections.values():
        if conn.enabled:
            input_key, output_key = conn.key
            start_pos = node_positions.get(input_key)
            end_pos = node_positions.get(output_key)
            
            if start_pos and end_pos:
                color = (0, 255, 0) if conn.weight > 0 else (255, 0, 0)
                line_width = min(4, max(1, int(abs(conn.weight) * 2)))
                pygame.draw.line(surface, color, start_pos, end_pos, line_width)

    # draw nodes
    for key, pos in node_positions.items():
        color = (200, 200, 200)
        if key in input_keys: color = (100, 100, 255)
        elif key in output_keys: color = (255, 100, 100)
        
        pygame.draw.circle(surface, color, pos, node_radius)
        pygame.draw.circle(surface, (0,0,0), pos, node_radius, 1)


        label_text = ""
        if key in input_keys:
            idx = input_keys.index(key)
            if idx < len(input_names): label_text = input_names[idx][:6] 
        elif key in output_keys:
            idx = output_keys.index(key)
            if idx < len(output_names): label_text = output_names[idx]
        else:
            label_text = str(key)
            
        if label_text:
            label_surface = font.render(label_text, True, (0,0,0))
            label_pos = (pos[0] + 12, pos[1] - 8)
            if key in input_keys:
                label_pos = (pos[0] - label_surface.get_width() - 12, pos[1] - 8)
            surface.blit(label_surface, label_pos)

def eval_genomes(genomes, config):

    global best_fitness_so_far
    
    space = pymunk.Space()
    space.gravity = (0, 1200) 

    spawn_pos = (150, SCREEN_HEIGHT - 200)

    ground = pymunk.Segment(space.static_body, (-2000, SCREEN_HEIGHT - 100), (20000, SCREEN_HEIGHT - 100), 8)
    ground.friction = 1.0
    ground.collision_type = 0
    space.add(ground)

    humanoids = []
    nets = []
    ge = []

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        humanoid = Humanoid(space, spawn_pos, i)
        humanoids.append(humanoid)
        ge.append(genome)

    running = True
    start_time = pygame.time.get_ticks()
    simulation_duration = 60000  
    wall_screen_x = spawn_pos[0] - 150  
    while running and len(humanoids) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if pygame.time.get_ticks() - start_time > simulation_duration:
            running = False

        current_wall_x = -10000
        for i, humanoid in enumerate(humanoids):
            if humanoid.is_alive:
                inputs = humanoid.get_inputs(current_wall_x)
                outputs = nets[i].activate(inputs)
                humanoid.apply_outputs(outputs)

        space.step(PHYSICS_STEPS)
        
        best_humanoid_this_gen = None
        current_max_fitness = -1e9
        
        alive_humanoids = [h for h in humanoids if h.is_alive]
        for humanoid in alive_humanoids:
            humanoid.check_fall(wall_screen_x)
            
            if humanoid.is_alive:
                humanoid.calculate_fitness()
                original_index = humanoids.index(humanoid)
                ge[original_index].fitness = humanoid.fitness

                if humanoid.fitness > current_max_fitness:
                    current_max_fitness = humanoid.fitness
                    best_humanoid_this_gen = humanoid
                    if humanoid.fitness > best_fitness_so_far:
                        best_fitness_so_far = humanoid.fitness

        for i in range(len(humanoids) - 1, -1, -1):
            if not humanoids[i].is_alive:
                humanoids[i].remove_from_space()
                humanoids.pop(i)
                nets.pop(i)
                ge.pop(i)
        
        if not humanoids:
             running = False

        screen.fill((135, 206, 235)) # sky col
        
        # Camera follow
        camera_x = 0
        if best_humanoid_this_gen:
            camera_x = max(0, best_humanoid_this_gen.bodies['torso'].position.x - SCREEN_WIDTH / 3)

        draw_options.transform = pymunk.Transform.translation(-camera_x, 0)
        space.debug_draw(draw_options)

        death_wall_speed = 5
        death_wall_x = spawn_pos[0] - 150 + ((pygame.time.get_ticks() - start_time) / 1000.0) * death_wall_speed
        wall_screen_x = death_wall_x
        pygame.draw.line(screen, (255, 0, 0), (wall_screen_x, 0), (wall_screen_x, SCREEN_HEIGHT), 3)

        # UI
        best_genome_to_draw = None
        best_fitness_in_gen = -float('inf')

        if ge and humanoids:
            best_idx = -1
            for i in range(len(ge)):
                if humanoids[i].is_alive and ge[i].fitness > best_fitness_in_gen:
                    best_fitness_in_gen = ge[i].fitness
                    best_idx = i
            if best_idx != -1:
                best_genome_to_draw = ge[best_idx]

        if best_genome_to_draw:
            nn_rect = pygame.Rect(SCREEN_WIDTH - 420, 20, 400, 350)
            pygame.draw.rect(screen, (240, 240, 240), nn_rect)
            pygame.draw.rect(screen, (0, 0, 0), nn_rect, 2)
            draw_neural_network(screen, best_genome_to_draw, config, (nn_rect.x + 10, nn_rect.y + 10), nn_rect.width - 20, nn_rect.height - 20)

        stats_y = 380
        info_texts = [
            f"Generation: {p.generation}",
            f"Best Fitness Ever: {best_fitness_so_far:.1f}",
            f"Current Best: {best_fitness_in_gen:.1f}",
            f"Alive: {len(humanoids)} / {len(genomes)}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (SCREEN_WIDTH - 200, stats_y + i * 25))

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    for humanoid in humanoids:
        humanoid.remove_from_space()

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    global p
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 1000)

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
