SAC (Soft Actor-Critic)
=======================

API Reference
-------------

.. autoclass:: modularl.agents.sac.SAC
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Here's an example of how to use the SAC agent:

.. code-block:: python

    import gym
    from modularl.agents.sac import SAC
    
    # Create an environment
    env = gym.make('Pendulum-v1')
    
    # Initialize the SAC agent
    agent = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=256,
        learning_rate=3e-4
    )
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Save the trained agent
    agent.save('sac_agent.pth')

This example demonstrates how to:

1. Create an environment
2. Initialize the SAC agent
3. Train the agent for a specified number of episodes
4. Use the agent to select actions and update its policy
5. Save the trained agent

You can adjust the hyperparameters and environment as needed for your specific use case.