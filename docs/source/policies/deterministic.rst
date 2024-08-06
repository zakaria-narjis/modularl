Deterministic Policy
====================

Deterministic Policy Class
--------------------------

.. autoclass:: modularl.policies.DeterministicPolicy
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Here's an example of how to use the DeterministicPolicy:

.. code-block:: python

    import torch
    import torch.nn as nn
    from modularl.policies.deterministic_policy import DeterministicPolicy

    # Define custom network
    class CustomNetwork(nn.Module):
        def __init__(self, observation_shape):
            super(CustomNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(observation_shape, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Observation and action space dimensions
    observation_shape = 10
    action_shape = 2
    high_action = 1.0
    low_action = -1.0

    # Custom network instance
    custom_network = CustomNetwork(observation_shape)

    # Create Deterministic Policy with the custom network
    policy = DeterministicPolicy(
        observation_shape=observation_shape,
        action_shape=action_shape,
        high_action=high_action,
        low_action=low_action,
        network=custom_network
    )

    # Or without a custom network
    default_policy = DeterministicPolicy(
        observation_shape=observation_shape,
        action_shape=action_shape,
        high_action=high_action,
        low_action=low_action
    )

    # Example observation
    observation = torch.randn((1, observation_shape))

    # Get action from the policy
    action = policy.get_action(observation)
    print("Action:", action)
