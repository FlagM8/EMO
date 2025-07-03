import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

bounds = [-15,15]
n_particles = 50
dim = 2
max_iter = 100

def fitness_function(x):
    return -np.sum(x**2-3*x+2) + np.sum(np.sin(x)) + 1

particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
velocities = np.random.uniform(-1, 1, (n_particles, dim))
personal_best_positions = np.copy(particles)
personal_best_scores = np.array([fitness_function(p) for p in particles])
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]


def update_velocity(particle, velocity, personal_best_position, global_best_position):
    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.0
    r1 = np.random.rand(dim)
    r2 = np.random.rand(dim)
    cognitive_velocity = cognitive_weight * r1 * (personal_best_position - particle)
    social_velocity = social_weight * r2 * (global_best_position - particle)
    new_velocity = (inertia_weight * velocity) + cognitive_velocity + social_velocity
    return new_velocity

def update_position(particle, velocity):
    new_position = particle + velocity
    new_position = np.clip(new_position, bounds[0], bounds[1])
    return new_position

# for iteration in range(max_iter):
#     for i in range(n_particles):
#         velocities[i] = update_velocity(particles[i], velocities[i], personal_best_positions[i], global_best_position)
#         particles[i] = update_position(particles[i], velocities[i])
#         score = fitness_function(particles[i])

#         if score < personal_best_scores[i]:
#             personal_best_scores[i] = score
#             personal_best_positions[i] = particles[i]

#     global_best_position = personal_best_positions[np.argmin(personal_best_scores)]


fig, ax = plt.subplots()
scat = ax.scatter([], [], c='blue', label='Particles')
best_dot = ax.scatter([], [], c='red', label='Global Best', marker='x')
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[0], bounds[1])
ax.set_title("PSO Optimization")
ax.legend()


scat = ax.scatter(particles[:, 0], particles[:, 1], c='blue', label='Particles')
best_dot = ax.scatter(global_best_position[0], global_best_position[1], c='red', label='Global Best', marker='x')
def init():
    global_best_score = np.min(personal_best_scores)
    scat.set_offsets(particles)
    best_dot.set_offsets(global_best_position)
    return scat, best_dot

def update(frame):
    global particles, velocities, personal_best_positions, personal_best_scores, global_best_position

    for i in range(n_particles):
        velocities[i] = update_velocity(particles[i], velocities[i], personal_best_positions[i], global_best_position)
        particles[i] = update_position(particles[i], velocities[i])
        score = fitness_function(particles[i])
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = particles[i]

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    scat.set_offsets(particles)
    best_dot.set_offsets(global_best_position)
    ax.set_title(f"Iteration {frame+1}, Best = {global_best_position}")
    return scat, best_dot


anim = FuncAnimation(fig, update, frames=max_iter, init_func=init, interval=100)
plt.show()