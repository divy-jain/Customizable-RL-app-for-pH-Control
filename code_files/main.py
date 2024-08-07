from simulation import SimKeys
from simulation import Simulation
from simulation import SimConfig
from export import Data
from control_ai import Agent, AgentConfig


def run_simulation(s_config: SimConfig, agent_configs):
    sim_time = 200_000
    sim_steps = int(sim_time / s_config.dt)

    for agent_config in agent_configs:
        sim = Simulation(s_config)
        agent = Agent(agent_config, SimKeys.ACTION, SimKeys.VALUE)
        data = Data(plots_every=50 * pow(10, 3), agent_config=agent_config, sim_config=s_config)

        target, time = sim.reset()

        for _ in range(sim_steps):
            # Change reference here if needed#
            ##################################
            
            action, predictions = agent.choose_action(target, time, sim.current_u)

            if predictions is not None:
                # Calculate actual trajectories
                values = sim.calculate_strategies(agent.get_allowed_strategies(sim.current_u), agent.h,
                                                  agent.act_every)
                data.record_errors(predictions, values, time)

            zi = 0
            # Add disturbance impulse z here if needed #
            ############################################
            target_, time, temp = sim.step(action, zi, agent.reference)
            if action is not None:
                agent.record(temp)
            agent.train_model(time)
            target = target_
        data.finish(sim)


def get_agent_configs():
    # Create all the Configs which should be compared

    c1 = AgentConfig(reference=8, action_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    c1.n = 4
    c1.m = 2
    c1.h = 4
    c1.act_every = 200
    c1.train_every = 10_000

    return [c1]


if __name__ == '__main__':
    sim_config = SimConfig()
    sim_config.noise = 0.5
    sim_config.dt = 10  # Time in ms
    sim_config.start_value = 7
    sim_config.ks = 0.15
    sim_config.stable_value = 7

    run_simulation(sim_config, get_agent_configs())
