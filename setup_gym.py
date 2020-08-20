from modules.robots import *
from modules.environments import *
from modules.agents import *
from modules.gym import Gym

AGENTS_COUNT = 16
LAYERS_DIMS = [64, 64, 64, 48]

Robot = withTimer(withMemory(withLastAction(ServAnt)))
# Robot = withTimer(ServAnt)

env = RobotsEnvironment([Robot(base_position=(((i % 4) - 2) * 5, ((i // 4) - 2) * 5, 0.5)) for i in range(AGENTS_COUNT)])

agents = [DenseAgent(
    len(env.observation_space.low[0].flatten()),
    len(env.action_space.low[0].flatten()),
    layer_dims=LAYERS_DIMS,
) for _ in range(AGENTS_COUNT)]
model = AgentsModel(agents)

fdim = 'x'.join(map(str, LAYERS_DIMS)) + '-' + str(AGENTS_COUNT)

gym = Gym(env, agents, model, fdim)
