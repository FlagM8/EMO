# Důležité části kódu evoluční simulace v `game_neat.py`

Tento dokument popisuje konkrétní části kódu, které jsou klíčové pro evoluční simulaci a využití NEAT v projektu. Zaměřuje se na práci s NEAT, evaluaci jedinců a konstrukci humanoidů.

---

## 1. Inicializace a běh NEAT

```python
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
```
- Hlavní vstupní bod programu. Spustí evoluci s konfigurací NEAT.

```python
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
```
- Vytvoření NEAT populace a spuštění evoluce - `eval_genomes`.

---

## 2. Evaluace jedinců (funkce `eval_genomes`)

```python
def eval_genomes(genomes, config):
    ...
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        humanoid = Humanoid(space, spawn_pos, i)
        humanoids.append(humanoid)
        ge.append(genome)
```
- Pro každý genom vytvoří neuronovou síť a instanci "humanoida".

```python
while running and len(humanoids) > 0:
    ...
    for i, humanoid in enumerate(humanoids):
        if humanoid.is_alive:
            inputs = humanoid.get_inputs(current_wall_x)
            outputs = nets[i].activate(inputs)
            humanoid.apply_outputs(outputs)
    ...
    for humanoid in alive_humanoids:
        humanoid.check_fall(wall_screen_x)
        if humanoid.is_alive:
            humanoid.calculate_fitness()
            original_index = humanoids.index(humanoid)
            ge[original_index].fitness = humanoid.fitness
```
- V každém kroku simulace se získají vstupy, neuronová síť je zpracuje a výstupy ovládají klouby "humanoida".
- Po každém kroku se kontroluje, zda "humanoid" nespadl nebo není za *death wall*
- Fitness se průběžně aktualizuje podle výkonu v simulaci.

---

## 3. Konstrukce a řízení humanoida

### Třída `Humanoid`

#### Konstrukce těla
```python
class Humanoid:
    def __init__(self, space, position, collision_type_offset):
        ...
        self.create_body(position)
        ...

def create_body(self, pos):
    ...
    self.bodies['torso'] = pymunk.Body(15, 150)
    self.bodies['torso'].position = (x, y - 50)
    ...
    self.bodies['head'] = pymunk.Body(5, 50)
    self.bodies['head'].position = (x, y - 110)
    ...
    for i in range(2):
        ...
        self.bodies[f'upper_leg_{i}'] = pymunk.Body(6, 100)
        ...
        self.bodies[f'lower_leg_{i}'] = pymunk.Body(4, 80)
        ...
        self.bodies[f'foot_{i}'] = pymunk.Body(2, 20)
        ...
    ...
    # Klouby a motory
    self.joints['head_joint'] = pymunk.PivotJoint(...)
    ...
    for i in range(2):
        self.joints[f'hip_joint_{i}'] = pymunk.PivotJoint(...)
        self.joints[f'hip_motor_{i}'] = pymunk.SimpleMotor(...)
        self.joints[f'hip_limit_{i}'] = pymunk.RotaryLimitJoint(...)
        ...
        self.joints[f'knee_joint_{i}'] = pymunk.PivotJoint(...)
        self.joints[f'knee_motor_{i}'] = pymunk.SimpleMotor(...)
        self.joints[f'knee_limit_{i}'] = pymunk.RotaryLimitJoint(...)
        ...
        self.joints[f'ankle_joint_{i}'] = pymunk.PivotJoint(...)
        self.joints[f'ankle_motor_{i}'] = pymunk.SimpleMotor(...)
        self.joints[f'ankle_limit_{i}'] = pymunk.RotaryLimitJoint(...)
```
- Každý "humanoid" je složen z těla, hlavy, dvou nohou (každá má dvě části) a chodidel.
- Klouby a motory umožňují pohyb jednotlivých částí, limity zabraňují nepřirozeným pohybům (jen teoreticky).

#### Vstupy pro neuronovou síť
```python
def get_inputs(self, wall_x):
    ...
    inputs.append(self.bodies['torso'].angle / math.pi)
    inputs.append(self.bodies['torso'].angular_velocity / 10.0)
    ...
    for i in range(2):
        ... # úhly kloubů, kontakt chodidla se zemí
    ...
    inputs.append(head_height)
    inputs.append(distance)
    return inputs
```
- Vstupy zahrnují úhly, rychlosti, kontakty chodidel, výšku hlavy a ušlou vzdálenost.

#### Aplikace výstupů neuronové sítě
```python
def apply_outputs(self, outputs):
    ...
    if len(outputs) >= 6:
        self.joints['hip_motor_0'].rate = outputs[0] * motor_speed
        self.joints['knee_motor_0'].rate = outputs[1] * motor_speed
        self.joints['ankle_motor_0'].rate = outputs[2] * motor_speed
        self.joints['hip_motor_1'].rate = outputs[3] * motor_speed
        self.joints['knee_motor_1'].rate = outputs[4] * motor_speed
        self.joints['ankle_motor_1'].rate = outputs[5] * motor_speed
```
- Výstupy neuronové sítě přímo nastavují rychlosti motorů v kloubech.

#### Kontrola pádu a kolize
```python
def check_fall(self, wall_x=None):
    ...
    if (self.bodies['head'].position.y >= ground_level + margin or 
        self.bodies['torso'].position.y >= ground_level + margin):
        self.is_alive = False
        return
    ...
    if abs(self.bodies['torso'].angle) > math.pi/2:
        self.is_alive = False
        return
    ...
    if wall_x is not None:
        if self.bodies['torso'].position.x < wall_x - 15:
            self.is_alive = False
            return
```
- "Humanoid" je označen jako mrtvý, pokud spadne, překročí limity nebo ho dožene death wall.

#### Výpočet fitness
```python
def calculate_fitness(self):
    ...
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
```
- Fitness zohledňuje vzdálenost, rychlost, stabilitu, výšku, pohyb nohou, penalizace za špatné postoje a kontakt se zemí. -jen se musí zlepšít její vyváženost

---

## 4. Vizualizace neuronové sítě

```python
def draw_neural_network(surface, genome, config, position, width, height):
    ...
```
- Funkce vykreslí aktuální topologii neuronové sítě nejlepšího jedince v generaci.

---

## 5. Další poznámky
- Kód obsahuje i správu prostředí (pymunk, pygame), vizualizaci a statistiky.
- Pro detailní nastavení evoluce a neuronových sítí slouží konfigurační soubor `config-feedforward.txt`.
