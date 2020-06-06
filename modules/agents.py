from keras.models import Model
from keras.layers import Dense, Input

class BaseAgent:
  def __init__(self, input_dim, output_dim, layers=[]):
    tensor = Input(shape=(input_dim,))
    self.input = tensor
    self.layers = layers[:]
    self.layers.append(Dense(output_dim))
    for layer in self.layers:
      tensor = layer(tensor)
    self.output = tensor

class DenseAgent(BaseAgent):
  def __init__(self, *args, layer_dims=[]):
    layers = [Dense(dim) for dim in layer_dims]
    super(DenseAgent, self).__init__(*args, layers)

def AgentsModel(agents):
  return Model(
      inputs=[agent.input for agent in agents],
      outputs=[agent.output for agent in agents],
  )
