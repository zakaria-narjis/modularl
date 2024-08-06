SAC (Soft Actor-Critic)
=======================

Agent Class
-------------

.. autoclass:: modularl.agents.SAC
    :members:
    :undoc-members:
    :show-inheritance:
    
Example Usage
-------------

Here's an example of how to use the SAC agent:

.. code-block:: python

    import torch
    import gym
    from modularl.agents import SAC
    from modularl.policies import GaussianPolicy
    from modularl.q_functions import SAQNetwork
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
    actor = GaussianPolicy(obsevation_shape, action_shape, high_action, low_action,custom_network_actor)
    critic = SAQNetwork(observation_shape, action_shape,custom_network_critic)

    #Initialize the actor and critics optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # Initialize the replay buffer
    buffer_size = 100000
    replay_buffer = ReplayBuffer(buffer_size,sampling="random")

    # Optional: Initialize the tensorboard writer
    writer = SummaryWriter()
    # Initialize the SAC agent
    num_episodes = 10000
    learning_starts = 1000 
    agent = SAC(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        replay_buffer = replay_buffer,
        learning_starts=learning_starts, # Number of steps before starting training (the total agents steps get's incremented by 1 evert agent.observe() call)
        entropy_lr=3e-4,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        device="cuda:0",
        target_network_frequency=2,
        target_entropy=-action_shape,
        writer=writer,
    )
    agent.init()
    # Training loop
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        state = torch.tensor(state, dtype=torch.float32)
        while not done:
            
            action,_,_ = agent.act_train(state)
            next_state, reward, done, _ = env.step(action)
            # Convert to torch tensors          
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            # Update the agent
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        writer.add_scalar('reward', episode_reward, episode)
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Save the trained agent
    agent.save('sac_agent.pth') #Not yet implemented

This example demonstrates how to:

1. Create a Gym environment (Pendulum-v1 in this case)
2. Extract necessary information from the environment (observation shape, action shape, and action bounds)
3. Define custom networks for the actor and critic (optional)
4. Initialize the GaussianPolicy (actor) and SAQNetwork (critic) with appropriate parameters
5. Set up optimizers for the actor and critic
6. Create a replay buffer
7. Initialize the SAC agent with all components and hyperparameters
8. Run a training loop to interact with the environment and update the agent
9. Log rewards using TensorBoard