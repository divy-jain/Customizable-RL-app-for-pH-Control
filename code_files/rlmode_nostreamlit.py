import numpy as np
import pandas as pd
from collections import deque, namedtuple
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import random
import io


# Define initial state and configuration parameters
initial_state = {
    'Rel. Time (in s)': 3000,
    'Fällungslauge.TotalVolume': 1.0,
    'Metallsulfatlösung.TotalVolume': 1.0,
    'R': 200,
    'RT': 210,
    'Tr': 39,
    'Vr': 300,
    'NH3 concentration in reactor (calculated)': 0.03,
    'Cges,MeSO4': 2,
    'CNH3': 0.25,
    'CNAOH': 5,
    'pH': 8.2  # Initial pH value
}


# Load the trained pH prediction model
with open('improved_model.pkl', 'rb') as model_file:
    ph_model = pickle.load(model_file)
state_scaler = pickle.load(open('scaler.pkl', 'rb'))


# Load the Polynomial GBR model and its scaler
with open('gbr_poly_model_final.pkl', 'rb') as model_file:
    poly_gbr_model = pickle.load(model_file)


with open('poly_scaler.pkl', 'rb') as scaler_file:
    poly_gbr_scaler = pickle.load(scaler_file)


poly = PolynomialFeatures(degree=2, include_bias=False)


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0


    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]


        probs = prios ** alpha
        probs /= probs.sum()


        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]


        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)


        return samples, indices, weights


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_dim, output_dim, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.memory = PrioritizedReplayBuffer(10000)
        self.last_action_moved_away = False
        self.last_action_type = None
        self.consecutive_negative_rewards = 0
        self.batch_size = 32


    def build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.input_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.output_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def choose_action(self, state, current_pH, desired_pH, tolerance=0.1):
        if abs(current_pH - desired_pH) <= tolerance:
            print("Within desired pH range, no adjustment needed.")
            self.consecutive_negative_rewards = 0  # Reset counter
            return np.array([0.0, 0.0])  # No action


        explore_probability = self.epsilon
        if self.consecutive_negative_rewards > 5:
            explore_probability = max(self.epsilon, min(0.5, self.consecutive_negative_rewards * 0.1))


        if np.random.rand() <= explore_probability:
            action1 = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0])
            action2 = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0])
            action = np.array([action1, action2])
            print("Exploring")
        else:
            q_values = self.model.predict(state)[0]
            action_index = np.argmax(q_values)
            action1 = ((action_index // 5) + 1) * 0.2
            action2 = ((action_index % 5) + 1) * 0.2
            action = np.array([action1, action2])
            print("Exploiting")


        print("Action chosen:", action)
        return action


    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])


        target_f = self.model.predict(state)
        action_index = ((action[0] / 0.2) - 1) * 5 + (action[1] / 0.2) - 1
        action_index = int(action_index)
        target_f[0][action_index] = target


        self.model.fit(state, target_f, epochs=1, verbose=0)


        if reward < 0:
            self.consecutive_negative_rewards += 1
        else:
            self.consecutive_negative_rewards = 0


    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


    def replay(self):
        if len(self.memory) < self.batch_size:
            return


        minibatch = random.sample(self.memory.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)


        states = np.array(states)
        next_states = np.array(next_states)


        # Reshape states and next_states to be 2D arrays
        states = states.reshape(self.batch_size, -1)
        next_states = next_states.reshape(self.batch_size, -1)


        # Get Q values for current states
        current_q_values = self.model.predict(states)
        # Get Q values for next states using target network
        next_q_values = self.target_model.predict(next_states)


        # Update Q values
        for i in range(self.batch_size):
            action_index = int(((actions[i][0] / 0.2) - 1) * 5 + (actions[i][1] / 0.2) - 1)
            if dones[i]:
                current_q_values[i][action_index] = rewards[i]
            else:
                current_q_values[i][action_index] = rewards[i] + self.gamma * np.max(next_q_values[i])


        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)


        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def predict_pH(current_state_array):
    current_state_array = np.reshape(current_state_array, (1, -1))
    scaled_features = state_scaler.transform(current_state_array)
    predicted_online_pH = ph_model.predict(scaled_features)[0][0]
    print(f"Predicted Online pH: {predicted_online_pH}")
    current_state_array_with_online_Ph = np.append(current_state_array, predicted_online_pH).reshape(1, -1)
    scaled_current_state = poly.fit_transform(current_state_array_with_online_Ph)
    scaled_current_state = poly_gbr_scaler.transform(scaled_current_state)
    predicted_offline_pH = poly_gbr_model.predict(scaled_current_state)[0]
    print(f"Predicted Offline pH: {predicted_offline_pH}")
    return predicted_offline_pH


def predict_pH_online(current_state_array):
    current_state_array = np.reshape(current_state_array, (1, -1))
    scaled_features = state_scaler.transform(current_state_array)
    predicted_online_pH = ph_model.predict(scaled_features)[0][0]
    print(f"Predicted Online pH: {predicted_online_pH}")
    return predicted_online_pH


def calculate_reward(online_after, desired_pH, online_before, offline_after, offline_before):
    # Calculate changes in online and offline pH
    online_delta = online_after - online_before
    offline_delta = offline_after - offline_before


    # Determine if we're moving in the right direction
    right_direction = (
        (online_delta > 0 and online_before < desired_pH) or
        (online_delta < 0 and online_before > desired_pH) or
        (online_delta == 0 and abs(online_before - desired_pH) < 0.1)
    )


    # Base reward on direction and magnitude of online change
    if right_direction:
        reward = 100 + abs(online_delta) * 1000
    else:
        reward = -100 - abs(online_delta) * 1000


    # Add a component based on overall proximity to desired offline pH
    distance_penalty = -abs(online_after - desired_pH) * 50


    # Bonus for being very close to desired pH
    bonus = 100 if abs(online_after - desired_pH) < 0.1 else 0


    # Penalty for large disagreement between online and offline predictions
    # prediction_agreement_penalty = -abs(online_delta - offline_delta) * 200


    total_reward = reward + distance_penalty + bonus


    return total_reward




def control_loop(agent, initial_state, data_df, desired_pH):
    time_step_interval = 4
    timesteps = 0
    batch_size = 10
    current_state = initial_state.copy()
    tolerance = 0.1


    metal_vol = 1.0
    fall_vol = 1.0


    results = {
        'time_step_list': [],
        'predicted_pH_list': [],
        'reward_list': [],
        'actual_pH_list': [],
        'predicted_online_pH': [],
        'fallvol': [],
        'metalvol': []
    }


    total_steps = len(data_df.iloc[0:1200])


    for index, row in data_df.iloc[22500:27800].iterrows():
            print(f"=== Time Step {index + 1}/{total_steps} ===")
            current_state = row.to_dict()
            results['actual_pH_list'].append(current_state['pH'])


            current_state['Fällungslauge.TotalVolume'] = fall_vol
            current_state['Metallsulfatlösung.TotalVolume'] = metal_vol


            current_time = current_state['Rel. Time (in s)']


            current_state_array = np.array([
                current_state['Rel. Time (in s)'],
                current_state['Fällungslauge.TotalVolume'],
                current_state['Metallsulfatlösung.TotalVolume'],
                current_state['R'],
                current_state['RT'],
                current_state['Tr'],
                current_state['Vr'],
                current_state['NH3 concentration in reactor (calculated)'],
                current_state['Cges,MeSO4'],
                current_state['CNH3'],
                current_state['CNAOH'],
            ])


            predicted_pH_before_action = predict_pH(current_state_array)
            predict_online = predict_pH_online(current_state_array)


            current_state_array = np.append(current_state_array, predict_online)
            current_state_array_reshaped = np.reshape(current_state_array, (1, -1))


            action = agent.choose_action(current_state_array_reshaped, predict_online, desired_pH, tolerance)


            if np.all(action == 0):
                results['predicted_online_pH'].append(predict_online)
                results['predicted_pH_list'].append(predicted_pH_before_action)
                results['reward_list'].append(0)
                results['time_step_list'].append(current_time)
                results['fallvol'].append(fall_vol)
                results['metalvol'].append(metal_vol)
            else:
                fall_vol += action[0]
                metal_vol += action[1]


                current_state['Fällungslauge.TotalVolume'] = fall_vol
                current_state['Metallsulfatlösung.TotalVolume'] = metal_vol


                updated_state_array = np.array([
                    current_time,
                    current_state['Fällungslauge.TotalVolume'],
                    current_state['Metallsulfatlösung.TotalVolume'],
                    current_state['R'],
                    current_state['RT'],
                    current_state['Tr'],
                    current_state['Vr'],
                    current_state['NH3 concentration in reactor (calculated)'],
                    current_state['Cges,MeSO4'],
                    current_state['CNH3'],
                    current_state['CNAOH'],
                ])


                predicted_pH_after_action = predict_pH(updated_state_array)
                online = predict_pH_online(updated_state_array)


                updated_state_array = np.append(updated_state_array, online)


                reward = calculate_reward(online, desired_pH, predict_online,predicted_pH_after_action,predicted_pH_before_action)


                agent.last_action_moved_away = reward < 0


                done = abs(online - desired_pH) <= tolerance


                agent.remember(current_state_array_reshaped, action, reward, np.reshape(updated_state_array, (1, -1)), done)


                agent.replay()


                if timesteps % 200 == 0:
                    agent.update_target_model()


                current_state['pH'] = predicted_pH_after_action


                results['predicted_online_pH'].append(online)
                results['predicted_pH_list'].append(predicted_pH_after_action)
                results['reward_list'].append(reward)
                results['time_step_list'].append(current_time)
                results['fallvol'].append(fall_vol)
                results['metalvol'].append(metal_vol)


            timesteps += 1

    calculate_statistics(results)
    return results

def calculate_statistics(results):
    actual_pH = np.array(results['actual_pH_list'])
    predicted_pH = np.array(results['predicted_pH_list'])
    predicted_online_pH = np.array(results['predicted_online_pH'])
    fall_vol = np.array(results['fallvol'])
    metal_vol = np.array(results['metalvol'])
    rewards = np.array(results['reward_list'])

    mean_actual_pH = np.mean(actual_pH)
    median_actual_pH = np.median(actual_pH)
    std_actual_pH = np.std(actual_pH)

    mean_online_pH = np.mean(predicted_online_pH)
    median_online_pH = np.median(predicted_online_pH)
    std_online_pH = np.std(predicted_online_pH)

    mean_predicted_pH = np.mean(predicted_pH)
    median_predicted_pH = np.median(predicted_pH)
    std_predicted_pH = np.std(predicted_pH)

    mae = np.mean(np.abs(predicted_online_pH - actual_pH))
    rmse = np.sqrt(np.mean((predicted_online_pH - actual_pH) ** 2))

    mean_fall_vol = np.mean(fall_vol)
    median_fall_vol = np.median(fall_vol)
    std_fall_vol = np.std(fall_vol)

    mean_metal_vol = np.mean(metal_vol)
    median_metal_vol = np.median(metal_vol)
    std_metal_vol = np.std(metal_vol)

    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)

    stats = {
        'mean_actual_pH': mean_actual_pH,
        'median_actual_pH': median_actual_pH,
        'std_actual_pH': std_actual_pH,
        'mean_predicted_pH': mean_predicted_pH,
        'median_predicted_pH': median_predicted_pH,
        'std_predicted_pH': std_predicted_pH,
        'mean_online_pH': mean_online_pH,
        'median_onlinepH': median_online_pH,
        'std_online_pH': std_online_pH,
        'mae': mae,
        'rmse': rmse,
        'mean_fall_vol': mean_fall_vol,
        'median_fall_vol': median_fall_vol,
        'std_fall_vol': std_fall_vol,
        'mean_metal_vol': mean_metal_vol,
        'median_metal_vol': median_metal_vol,
        'std_metal_vol': std_metal_vol,
        'mean_reward': mean_reward,
        'median_reward': median_reward,
        'std_reward': std_reward
    }

    print(stats)

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))


    ax1.plot(results['time_step_list'], results['predicted_pH_list'], marker='o', linestyle='-', color='b', label='Predicted pH')
    ax1.plot(results['time_step_list'], results['actual_pH_list'], marker='x', linestyle='--', color='g', label='Actual pH')
    ax1.plot(results['time_step_list'], results['predicted_online_pH'], marker='^', linestyle=':', color='r', label='Predicted Online pH')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('pH')
    ax1.legend(loc='upper left')
    ax1.set_title('pH over Time')


    ax2.plot(results['time_step_list'], results['fallvol'], marker='o', linestyle='-', color='c', label='Fällungslauge Volume')
    ax2.plot(results['time_step_list'], results['metalvol'], marker='s', linestyle='-', color='y', label='Metallsulfatlösung Volume')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.set_title('Reagent Volumes over Time')


    plt.tight_layout()
    plt.show()


def main():
    print("pH Control Simulation")


    data_df = pd.read_excel("C:\\Users\\IPAT2024\\Desktop\\Divyansh\\ReactorData\\Control-pH-Model_Data-main.xlsx")
    print("File successfully uploaded and read!")


    print("\nData Preview:")
    print(data_df.head())


    desired_pH = float(input("Enter desired pH (0.0 to 14.0): "))


    initial_state = {
        'Rel. Time (in s)': 3000,
        'Fällungslauge.TotalVolume': 1.0,
        'Metallsulfatlösung.TotalVolume': 1.0,
        'R': 200,
        'RT': 210,
        'Tr': 39,
        'Vr': 300,
        'NH3 concentration in reactor (calculated)': 0.03,
        'Cges,MeSO4': 2,
        'CNH3': 0.25,
        'CNAOH': 5,
        'pH': 8.2
    }


    agent = DQNAgent(input_dim=12, output_dim=25)


    # Run the control loop
    results = control_loop(agent, initial_state, data_df, desired_pH)


    print("Simulation completed!")
    plot_results(results)


if __name__ == "__main__":
    main()
