import numpy as np
from scipy.integrate import odeint


class SimKeys:          #defines some keys used to access the simulation data
    TIME = "time"
    VALUE = "value"
    CHANGE = "change"
    DELTA = "delta"
    ACTION = "action"
    REFERENCE = "reference"
    K1 = "k1"
    K2 = "k2"
    Z = "z"


#The k1_function(t), k2_function(t), and zd_function(t) functions are used to define the time-varying parameters of the system, such as the coefficient values k1 and k2 and the disturbance signal zd. These functions can be modified to define different system configurations.
def k1_function(t):
    # if t < 100:
    #     return 0.2
    # elif t < 200:
    #     return 0.4 - 0.2 * math.exp(-(t - 100) / 5)
    # else:
    #     return 0.2 + 0.2 * math.exp(-(t - 200) / 2)

    #return 0.1 * math.sin(0.2*t) + 0.2
    return 0.2


def k2_function(t):
    # if t < 100:
    #     return 0.05
    # elif t < 200:
    #     return 0.5 - 0.45 * math.exp(-(t - 100) / 10)
    # else:
    #     return 0.05 + 0.45 * math.exp(-(t - 200) / 2)

    #return 0.2 * math.cos(0.5*t) + 0.3
    return 0.05


def zd_function(t):
    #return 0.2 * math.sin(0.1 * t)
    return 0


#The SimConfig class is used to store the configuration parameters of the simulation, such as the time step dt, the minimum and maximum values of the system, the stable value of the system, the start value of the system, and the noise level. It also includes the functions defined above for k1, k2, and zd. The SimConfig object is passed to the Simulation object to initialize the simulation
class SimConfig:
    # dt in ms
    def __init__(self):
        self.dt = 10
        self.k1 = lambda t: k1_function(t)
        self.k2 = lambda t: k2_function(t)
        self.zd = lambda t: zd_function(t)
        self.ks = 0.15
        self.min_value = 0
        self.max_value = 14
        self.stable_value = 7
        self.start_value = 7
        self.noise = 0.5
#The Simulation class is the main class responsible for simulating the system. It takes a SimConfig object as a parameter and initializes the parameters of the simulation. 
class Simulation:
    def __init__(self, simulation_config: SimConfig):
        self.dt = simulation_config.dt
        self.start_val = simulation_config.start_value
        self.current_val = simulation_config.start_value
        self.stable_val = simulation_config.stable_value
        self.k1 = simulation_config.k1
        self.k2 = simulation_config.k2
        self.ks = simulation_config.ks
        self.zd = simulation_config.zd
        self.min_val = simulation_config.min_value
        self.max_val = simulation_config.max_value
        self.noise = simulation_config.noise
        self.df = []
        self.current_t = 0
        self.current_u = 0

#The step function is used to calculate the next state of the system, given the current input u, the disturbance signal zi, and the reference value w.
    def step(self, u, zi, w):
        if u is not None:
            self.current_u = u #If a new input value u is provided, it updates the current_u attribute of the Simulation object to reflect the new input.

        # Calculate disturbance :The function calculates the disturbance signal z as the sum of three components: a normally distributed noise term with mean zero and standard deviation noise, the disturbance input zi, and the time-varying disturbance signal zd defined in the SimConfig object.
        noise = np.random.normal(0, self.noise, 1)
        z = noise[0] + zi + self.zd(self.current_t / 1000)
        #The function calculates the change in the system state change based on the current parameters of the system and the input u and disturbance signal z. It does this by calling the _calculate_change function.
        change = self._calculate_change(self.current_u, z)


        # Old system state (before step)
        #The function creates a dictionary temp that contains the current time, the current system value, the change in the system value, the current input value, the reference value, the values of k1, k2, and z, and the disturbance signal z.
        temp = {
            SimKeys.TIME: self.current_t,
            SimKeys.VALUE: self.current_val,
            SimKeys.DELTA: self.current_val - w,
            SimKeys.CHANGE: change,
            SimKeys.ACTION: self.current_u,
            SimKeys.REFERENCE: w,
            SimKeys.K1: self.k1(self.current_t / 1000),
            SimKeys.K2: self.k2(self.current_t / 1000),
            SimKeys.Z: z
        }
        self.df.append(temp) #The function appends the temp dictionary to the df list, which stores the history of the system's behavior.

        self.current_val += change #The function updates the current value of the system by adding the change to the current value.

        # Cap value between min and max
        #The function checks whether the updated value is outside of the valid range [min_val, max_val] defined in the SimConfig object. If it is, it sets the updated value to the nearest valid value.
        if self.current_val < self.min_val:
            self.current_val = self.min_val
        elif self.current_val > self.max_val:
            self.current_val = self.max_val

        self.current_t += self.dt #The function updates the current time by adding the time step dt to the current time.

        #The function returns the updated system value, the updated time, and the temp dictionary.
        return self.current_val, self.current_t, temp



#The reset function is used to reset the simulation to its initial state.
    def reset(self):
        self.current_t = 0
        self.current_u = 0
        self.df.clear()
        self.current_val = self.start_val

        return self.current_val, self.current_t



#The _calculate_change function is used to calculate the change in the system state based on the current parameters of the system and the input u and disturbance signal z.
    def _calculate_change(self, u, z):
        # Divide by 1000 to convert from ms -> s
        return (-self.k1(self.current_t / 1000) * abs(self.stable_val - self.current_val) - self.k2(
            self.current_t / 1000) + self.ks * u + z) * self.dt / 1000



#The calculate_strategies function is used to calculate the actual influence of the different control strategies on the system. This function uses the odeint function from scipy.integrate to solve a system of ordinary differential equations that describe the system's behavior. 
    def calculate_strategies(self, strategies: np.array, p_horizon, act_every):
        # Calculate the actual influence of the different control strategies on the system
        def model(y, t, u, dt, t0):
            return -(self.k1(t0 + t) * abs(y - self.stable_val) + self.k2(t0 + t)) + self.ks * u[:, min(int(t / dt), u.shape[1] - 1)] + self.zd(t0 + t)

        delta_t = act_every / 1000
        t_eval = np.arange(0, p_horizon + 1, 1) * delta_t
        y0 = np.ones(shape=strategies.shape[0]) * self.current_val

        values = odeint(model, y0, t_eval, args=(strategies, delta_t, self.current_t / 1000))
        # Transpose matrix, because somehow it is in the wrong order
        values = np.array(values).T
        # Drop first column since it is the one for t=0 which we don't need
        values = values[:, 1:]

        return values



#The _calculate_change function in the Simulation class is used to calculate the change in the system state based on the current parameters of the system, the input u, and the disturbance signal z. Here's a breakdown of what the function does:

# #The function takes two arguments: the input u and the disturbance signal z.

# The function calculates the value of k1 and k2 at the current time by calling their corresponding functions with the argument self.current_t / 1000.

# The function calculates the absolute difference between the current value of the system self.current_val and the stable value of the system self.stable_val.

# The function calculates the change in the system value change based on the following formula:

# change = (-k1 * |current_val - stable_val| - k2 + ks * u + z) * dt / 1000

# where k1 and k2 are the current values of the coefficients, ks is the steady-state gain of the system, u is the input value, z is the disturbance signal, and dt is the time step of the simulation. The function multiplies the whole expression by dt/1000 to convert from milliseconds to seconds.

# The function returns the calculated value of change.

# This function is used to calculate the change in the system value at each time step of the simulation. The expression for change is derived from the mathematical model of the system and takes into account the input, the disturbance, and the time-varying coefficients of the system. By modifying the expressions for k1 and k2, one can change the behavior of the system and study its response to different inputs and disturbances.





# The calculate_strategies function in the Simulation class is used to calculate the actual influence of different control strategies on the system. It takes three arguments: strategies, a numpy array that contains the control strategies to be evaluated, p_horizon, the prediction horizon (in milliseconds), and act_every, the frequency at which control actions are taken (in milliseconds). Here's a breakdown of what the function does:

# The function defines a new function model, which takes five arguments: y, the current state of the system; t, the current time; u, a matrix of control strategies; dt, the time step of the simulation; and t0, the starting time of the simulation. The function returns the rate of change of the system state based on the current system state, the control strategy, and the time-varying coefficients of the system.

# The function calculates the time step of the simulation delta_t by dividing the frequency of control actions act_every by 1000 to convert from milliseconds to seconds.

# The function creates a numpy array t_eval that contains the time points at which the system state will be evaluated. It does this by calling np.arange with the arguments (0, p_horizon + 1, 1) * delta_t, which creates a range of time values from 0 to p_horizon, incremented by delta_t at each step.

# The function creates a numpy array y0 that contains the initial state of the system. It initializes the array to a vector of ones with the same length as the number of control strategies, multiplied by the current value of the system.

# The function calls the odeint function from the scipy.integrate module to solve the differential equations that describe the behavior of the system. It passes model as the function to be solved, y0 as the initial conditions, t_eval as the time points at which the solution will be evaluated, and the arguments (strategies, delta_t, self.current_t / 1000) as additional arguments to model.

# The function transposes the resulting array of system states values to put each control strategy in a separate row. It then drops the first column, which corresponds to the initial time t0=0 and which we don't need.

# The function returns the resulting numpy array values, which contains the actual influence of the different control strategies on the system over the prediction horizon.

# This function is used to evaluate the effectiveness of different control strategies on the system by simulating their impact over a certain time horizon. The output of the function can be used to select the most effective control strategy or to design new control strategies that optimize the behavior of the system.



