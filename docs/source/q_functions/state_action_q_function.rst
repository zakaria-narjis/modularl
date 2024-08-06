State-Action Q-Function
=======================

State-Action Q-Function Class
-----------------------------

.. autoclass:: modularl.q_functions.SAQNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Here is an example of how to create and use an instance of the `SAQNetwork` class:

.. code-block:: python
   
   import torch
   import torch.nn as nn
   from modularl.q_functions import SAQNetwork

   # Define the observation and action shapes
   observation_shape = 10
   action_shape = 2

   # Initialize the Q-network with default settings
   q_network = SAQNetwork(observation_shape, action_shape)

   # Example observation and action tensors
   observation = torch.randn(1, observation_shape)
   actions = torch.randn(1, action_shape)

   # Compute the Q-value
   q_value = q_network(observation, actions)
   print(f"Q-value: {q_value.item()}")
