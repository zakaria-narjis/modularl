DDPG (Deep Deterministic Policy Gradient)
=========================================

Agent Class
-----------

.. autoclass:: modularl.agents.DDPG
    :members:
    :undoc-members:
    :show-inheritance:
    
Example Usage
-------------

Here's an example of how to use the DDPG agent:

.. code-block:: python

    import torch
    import gym
    from modularl.agents import DDPG
    from modularl.policies import DeterministicPolicy
    from modularl.q_functions import QNetwork
    from modularl.replay_buffers import ReplayBuffer
    import copy

    # Create an environment
    env = gym.make('Pendulum-v1')

    # Get observation and action space dimensions
    observation_shape = env.observation_space.shape[0]  # 3 for Pendulum-v1
    action_shape = env.action_space.shape[0]  # 1 for Pendulum-v1

    # Get action bounds
    high_action = env.action_space.high[0]  # 2.0 for Pendulum-v1
    low_action = env.action_space.low[0]  # -2.0 for Pendulum-v1

    # Optional: Define a custom network
    class CustomNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return x

    custom_network_actor = CustomNetwork(observation_shape, 64)
    custom_network_critic = copy.deepcopy(custom_network_actor)

    # Initialize the actor and critic networks
    actor = DeterministicPolicy(observation_shape, action_shape, high_action, low_action, custom_network_actor)
    critic = QNetwork(observation_shape, action_shape, custom_network_critic)

    # Initialize the actor and critic optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Initialize the replay buffer
    buffer_size = 100000
    replay_buffer = ReplayBuffer(buffer_size)

    # Initialize the DDPG agent
    agent = DDPG(
        actor=actor,
        qf=critic,
        actor_optimizer=actor_optimizer,
        q_optimizer=critic_optimizer,
        replay_buffer=replay_buffer,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        exploration_noise=0.1,
        device="cuda:0",
    )
    agent.init()

    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        state = torch.tensor(state, dtype=torch.float32)
        while not done:
            action = agent.act_train(state)
            next_state, reward, done, _ = env.step(action)
            # Convert to torch tensors
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            # Observe and update the agent
            agent.observe(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Save the trained agent
    agent.save('ddpg_agent.pth') #Not implemented yet (consider saving the actor and critic models manually)

This example demonstrates how to:

1. Create a Gym environment (Pendulum-v1 in this case)
2. Extract necessary information from the environment (observation shape, action shape, and action bounds)
3. Define custom networks for the actor and critic (optional)
4. Initialize the DeterministicPolicy (actor) and QNetwork (critic) with appropriate parameters
5. Set up optimizers for the actor and critic
6. Create a replay buffer
7. Initialize the DDPG agent with all components and hyperparameters
8. Run a training loop to interact with the environment and update the agent
9. Log rewards using print statements (TensorBoard logging can be added similarly)

The main differences between DDPG and TD3 in this example are:

- DDPG uses a single critic (QNetwork) instead of two critics
- The hyperparameters are slightly different (e.g., tau, learning rates)
- DDPG doesn't have policy_noise, noise_clip, and policy_frequency parameters