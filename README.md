# Customizable-RL-app-for-pH-Control
Streamlit based application with customizable Reinforcement learning parameters, for adaptive pH control in battery engineering. Built for Institute for Particle Technology, Braunschweig. 

This project implements a Deep Q-Learning (DQL) agent for optimizing pH control in a chemical reactor. The solution leverages predictive modeling and reinforcement learning techniques to maintain the desired pH levels by adjusting reactant volumes.

## Table of Contents

1. [Predictive Modeling](#predictive-modeling)
   - [Neural Network for Online pH Prediction](#neural-network-for-online-ph-prediction)
   - [Gradient Boosting Regression for Offline pH Prediction](#gradient-boosting-regression-for-offline-ph-prediction)
2. [Reinforcement Learning Agent](#reinforcement-learning-agent)
   - [State Representation](#state-representation)
   - [Action Space](#action-space)
   - [Reward Function](#reward-function)
   - [Experience Replay and Prioritization](#experience-replay-and-prioritization)
   - [Deep Q-Learning Architecture](#deep-q-learning-architecture)
3. [Control Loop Simulation](#control-loop-simulation)
   - [Training the DQL Agent](#training-the-dql-agent)
   - [Control Loop Operation](#control-loop-operation)
4. [Streamlit App and customizability](#streamlit-app)
5. [Getting Started](#getting-started)
6. [License](#license)

## Predictive Modeling

### Neural Network for Online pH Prediction

A neural network model was developed to predict the online pH based on reactor conditions.

- **Model Architecture**:
  - **Input Layer**: Normalized features from the Control-pH-Model Data.
  - **Hidden Layers**: Three dense layers with 64, 32, and 16 neurons, respectively, with ReLU activation and dropout (rate: 0.2).
  - **Output Layer**: A single neuron with a linear activation function to predict the pH level.

- **Training and Evaluation**:
  - **Data Split**: 80% for training, 20% for evaluation.
  - **Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

### Gradient Boosting Regression for Offline pH Prediction

A Gradient Boosting Regressor (GBR) model with polynomial features was employed to predict offline pH.

- **Model Parameters**:
  - Number of Trees: 100
  - Learning Rate: 0.1
  - Maximum Depth: 3
  - Subsample: 0.8

- **Training and Evaluation**:
  - Trained on the T-Model-Reactor Data.
  - Evaluated using the same metrics as the neural network model.

## Reinforcement Learning Agent

### State Representation

The state for the DQL agent includes:

- **Reaction Time**: Elapsed time since the start of the reaction.
- **Volumes of Reactants**: Current volumes of reactants in the reactor.
- **Reactor Temperature**: Current reactor temperature.
- **Predicted pH Values**: pH levels predicted by the GBR model.

Additional polynomial features capture non-linear interactions between these variables.

### Action Space

The action space consists of discrete adjustments to the volumes of reactants, allowing exploration of various control strategies.

### Reward Function

The reward function aims to minimize deviation from the desired pH level by assigning positive rewards for actions that bring the pH closer to the target.

### Experience Replay and Prioritization

- **Experience Replay**: Stores past experiences (state, action, reward, next state) in a buffer for training.
- **Prioritized Experience Replay**: Samples experiences with higher temporal-difference (TD) errors more frequently.

### Deep Q-Learning Architecture

- **Input Layer**: Corresponds to the state representation, including reactor conditions and polynomial features.
- **Hidden Layers**: Three dense layers with 128, 64, and 32 neurons, with ReLU activation functions.
- **Output Layer**: Represents Q-values for each possible action.

Trained using the Adam optimizer (learning rate: 0.001) and MSE loss function.

## Control Loop Simulation

### Training the DQL Agent

- **Initialization**: Initialize experience replay buffer and neural network parameters.
- **Exploration and Exploitation**: Use decaying epsilon-greedy strategy.
- **Experience Collection**: Store and sample experiences for training.
- **Network Updates**: Periodically update the target Q-network.

### Control Loop Operation

1. **Data Collection**: Gather real-time data on reactor conditions.
2. **State Estimation**: Predict current pH using the GBR model.
3. **Decision Making**: Select the optimal action based on the DQL agent’s evaluation.
4. **Action Execution**: Adjust reactant volumes based on the chosen action.
5. **Feedback and Learning**: Measure resulting pH and update the agent’s policy.

## Streamlit Customizability

This Streamlit application allows for extensive customization to tailor the pH control simulation to your needs. Below is a guide on how to use the various features available in the Streamlit interface.

### Streamlit Interface

1. **Model Selection**
   - **Upload Models and Scalers**: In the sidebar, you can upload the following:
     - **pH Prediction Model**: For real-time pH prediction.
     - **Gradient Boosting Regressor (GBR) Model**: For offline pH prediction.
     - **State Scaler**: To normalize state features.
     - **Polynomial Scaler**: For polynomial features in GBR.

2. **Data Upload**
   - **Choose File**: Upload an Excel file containing your simulation data.

3. **Column Selection**
   - **Control Variables**: Select the control variables from your data.
   - **Action Ranges**: Specify the action ranges for each control variable.

4. **Reward Function Customization**
   - **Direction Rewards**: Adjust the rewards based on the direction of control actions.
   - **Magnitude Rewards**: Set rewards based on the magnitude of control actions.
   - **Proximity Rewards**: Configure rewards based on proximity to the target pH.
   - **Target Reached Bonus**: Set a bonus reward for reaching the desired pH level.

5. **Neural Network Customization**
   - **Hidden Layers and Neurons**: Define the number of hidden layers and neurons per layer for the Reinforcement Learning (RL) agent’s neural network.

6. **Hyperparameter Customization**
   - **Epsilon**: Set the exploration rate.
   - **Epsilon Decay**: Define how epsilon decreases over time.
   - **Learning Rate**: Adjust the learning rate for the RL algorithm.
   - **Gamma**: Set the discount factor for future rewards.

7. **Desired pH**
   - **Target pH Level**: Specify the desired pH level for the simulation.

8. **Data Range Selection**
   - **Start and End Indices**: Specify the range of indices to be used in the simulation.

9. **Run Simulation**
   - **Start Simulation**: Click the "Run Simulation" button to begin the pH control simulation and visualize the results.

### Results

- **Plots**: The application will display plots showing the predicted and actual pH levels, as well as reagent volumes over time.
- **Performance Metrics**: Metrics such as average reward and final pH difference are shown.
- **Download Results**: Download the simulation results as a CSV file for further analysis.

Feel free to explore these options to customize the simulation according to your needs and get the most out of your pH control experiments.


## Getting Started

   ```bash
   git clone https://github.com/yourusername/your-repository.git

   pip install -r requirements.txt

   streamlit run app.py


