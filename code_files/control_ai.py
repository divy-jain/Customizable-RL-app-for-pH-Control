import numpy as np
import utils as utils
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam


class AgentConfig:
# It stores configuration parameters for the agent, such as neural network architecture, training frequency, memory size, action space, and exploration-exploitation parameters.
    def __init__(self, reference, action_space, du_max=None, sys_lim=None):
        self.max_mem_size = 10_000
        self.n = 2
        # h needs to be greater than m and > 0, m can be 0
        # Control horizon
        self.m = 2
        # Prediction horizon
        self.h = 4
        self.epsilon = 1
        self.epsilon_dec = 0.7
        self.epsilon_min = 0.01
        self.batch_size = 10
        self.epochs = 50
        # Train model every 10 seconds
        self.train_every = 10_000
        # Act every 200ms
        self.act_every = 200
        self.alpha = 0.7
        self.reference = reference
        self.action_space = action_space
        self.du_max = du_max
        self.sys_lim = sys_lim


def build_network(input_size, output_size, n_hidden1, n_hidden2):
    model = Sequential([
        Dense(n_hidden1, input_shape=(input_size,)),
        LeakyReLU(alpha=0.1),
        Dense(n_hidden2),
        LeakyReLU(alpha=0.1),
        Dense(output_size),
        LeakyReLU(alpha=0.1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model

# This class defines the main logic of the agent. It includes methods for recording the environment state, choosing actions, training the model, and calculating the cost function.
class Agent:
    #Initializes the agent with the given configuration, creates the KNN model and computes all possible control strategies
    def __init__(self, config: AgentConfig, c_key, t_key):
        assert config.h > config.m
        self.input_size = (config.n + 1) * 2 + config.m
        # System limits (upper limit and lower limit) -> if not None the data will be normalized to that range
        self.sys_lim = config.sys_lim
        self.action_space = config.action_space
        self.du_max = config.du_max
        self.alpha = config.alpha
        self.epsilon = config.epsilon
        self.epsilon_dec = config.epsilon_dec
        self.epsilon_min = config.epsilon_min
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.reference = config.reference
        self.n = config.n
        self.control_key = c_key
        self.target_key = t_key
        self.m = config.m
        self.h = config.h
        self.train_every = config.train_every
        self.last_trained = 0
        self.act_every = config.act_every
        self.last_acted = 0
        self.memory = deque(maxlen=config.max_mem_size)
        self.training_history = []
        # Temporary memory is cleared after every training iteration
        self.temp_memory = []
        self.model = build_network(self.input_size, self.h, 256, 128)
        self.control_strategies = utils.compute_all_possible_strategies(self.m, self.action_space,
                                                                        du_max=self.du_max)#
        #Prepares the input data for the KNN to make predictions, normalizing the data if system limits are set
    def _get_prediction_data(self, current_y, allowed_strategies):
        # Get the target and control values for the last n time steps from the memory
        data = [[self.memory[i][self.target_key], self.memory[i][self.control_key]] for i in
                range(len(self.memory) - self.n, len(self.memory))]
        data = np.array(data, dtype=float)

        if self.sys_lim is not None:
            # Normalize Data to system limits
            data[:, 0] = utils.normalize(data[:, 0], self.sys_lim[0], self.sys_lim[1])
            data[:, 1] = utils.normalize(data[:, 1], min(self.action_space), max(self.action_space))
            allowed_strategies = utils.normalize(allowed_strategies, min(self.action_space), max(self.action_space))

        # Flatten array since its currently nested
        data = data.flatten()
        # Add current y at the end of the array
        if self.sys_lim is not None:
            data = np.append(data, utils.normalize(current_y, self.sys_lim[0], self.sys_lim[1]))
        else:
            data = np.append(data, current_y)

        # Expand dimensions of array so that the row can be copied
        data = np.expand_dims(data, axis=0)
        # Copy the row to fit the same shape as the control strategies so that they can be concatenated
        data = np.repeat(data, repeats=allowed_strategies.shape[0], axis=0)
        data = np.concatenate((data, allowed_strategies), axis=1)

        return data
#Filters control strategies based on the constraint du <= du_max.
    def get_allowed_strategies(self, current_u):
        if self.du_max is None:
            return self.control_strategies

        # Filter possible control strategies with constraint du <= du_max
        diff = np.abs(current_u - self.control_strategies[:, 0])
        return self.control_strategies[diff <= self.du_max]
#Records the current state of the environment in the agent's memory
    def record(self, state):
        # Save current target and control values in the long term and temporary memory
        inputs = [self.target_key, self.control_key]
        y = {k: state[k] for k in inputs}
        self.memory.append(y)
        self.temp_memory.append(y)
#Chooses the next action to take based on the current state and time, using the epsilon-greedy algorithm. It returns the best action and predictions made by the KNN.
    def choose_action(self, current_y, t, u):
        # Returns next action and all the predictions the ANN made
        if t < self.last_acted + self.act_every:
            return None, None

        self.last_acted = t
        rand = np.random.random()
        if len(self.memory) < self.n or rand < self.epsilon:
            # Pick random action
            action = np.random.choice(self.action_space)
            return action, None

        allowed_strategies = self.get_allowed_strategies(u)
        inputs = self._get_prediction_data(current_y, np.copy(allowed_strategies))
        predictions = self.model.predict(inputs, batch_size=4096)
        if self.sys_lim is not None:
            predictions = utils.denormalize(predictions, self.sys_lim[0], self.sys_lim[1])

        # Evaluate control strategies and find the best one (minimal cost)
        costs = _cost_function(predictions, self.reference, self.alpha)
        rand = np.random.random()
        if rand <= 0.1:
            # Pick a random strategy of the best 10
            idx = np.argpartition(costs, min(10, allowed_strategies.shape[0] - 1))
            min_cost_idx = np.random.choice(idx)
        else:
            min_cost_idx = np.argmin(costs)

        best_strategy = allowed_strategies[min_cost_idx]
        print('BEST STRATEGY: ', best_strategy, ' COST: ', costs[min_cost_idx], ' EPSILON: ', self.epsilon)

        # Only use first action of control strategy for the next timestep, after that repeat this process
        return best_strategy[0], predictions

    def train_model(self, t):
        if len(self.memory) < self.n + self.h + 1 + self.batch_size or t < self.last_trained + self.train_every:
            return

        df = pd.DataFrame.from_records(self.temp_memory)

        # Normalize data
        if self.sys_lim is not None:
            df[self.target_key] = utils.normalize(df[[self.target_key]], self.sys_lim[0], self.sys_lim[1])
            df[self.control_key] = utils.normalize(df[[self.control_key]], min(self.action_space),
                                                   max(self.action_space))

        x_data, y_data = utils.convert_input_data_training(df, self.n, self.m, self.h, self.control_key,
                                                           self.target_key)

        history = self.model.fit(x_data, y_data, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        self.training_history.append(history)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        self.temp_memory.clear()
        self.last_trained = t

#Calculates the cost of the predicted trajectories using the weighted sum of squared errors, considering a reference value and an alpha parameter for discounting future time steps.
def _cost_function(trajectories, reference, alpha):
    '''
       Calculate cost of trajectories
       :param trajectories: Predicted trajectories over a given horizon of len(trajectory)
       :param reference: Value where the system should end up
       :return: cost of all trajectories
    '''
    h = trajectories.shape[1]

    # Calculate weights for each element in the trajectory
    weights = np.power(alpha, np.flip(np.arange(h)))
    return np.sum((trajectories - reference) ** 2 * weights, axis=1)

