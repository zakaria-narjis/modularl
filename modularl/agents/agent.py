import abc
import torch


class AbstractAgent(abc.ABC):
    """
    Abstract base class for all agents.

    This class outlines the methods that an agent should implement to interact
    with an environment and learn from it.

    Methods
    -------
    init():
        Initialize the agent.
    observe(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones):
        Observe the environment and store the transition in the replay buffer.
    act_train(batch_obs):
        Select an action for training.
    act_eval(obs):
        Select an action for evaluation.
    update():
        Perform a training update.
    """  # noqa: E501

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the agent with the provided keyword arguments.

        :param kwargs: Keyword arguments for agent initialization.
        """  # noqa: E501

    @abc.abstractmethod
    def init(self):
        """
        Initialize the agent.
        """  # noqa: E501

    @abc.abstractmethod
    def observe(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_dones: torch.Tensor,
    ):
        """
        Observe the environment and store the transition in the replay buffer.

        :param batch_obs: (torch.Tensor) Tensor containing the observations.
            Shape: (batch_size, \\*)
        :param batch_actions: (torch.Tensor) Tensor containing the actions.
            Shape: (batch_size, action_dim)
        :param batch_rewards: (torch.Tensor) Tensor containing the rewards.
            Shape: (batch_size,)
        :param batch_next_obs: (torch.Tensor) Tensor containing the next observations.
            Shape: (batch_size, \\*)
        :param batch_dones: (torch.Tensor) Tensor containing the dones.
            Shape: (batch_size,)

        where:
            - batch_size is the number of transitions in the batch
            - action_dim is the dimension of the action space
            - \\* in the observation shape represents the dimension of the observation space, which depends on the environment and the networks (e.g., (3, 64, 64) for images or (512,) for vectors)

        Note:
            - The shapes assume a vectorized environment. For a single environment, batch_size would typically be 1.

        """  # noqa: E501

    @abc.abstractmethod
    def act_train(self, batch_obs: torch.Tensor) -> torch.Tensor:
        """
        Select an action for training.

        :param batch_obs: (torch.Tensor) Tensor containing the observations.
        :return: (torch.Tensor) Selected actions for training.
        """  # noqa: E501

    @abc.abstractmethod
    def act_eval(self, batch_obs: torch.Tensor) -> torch.Tensor:
        """
        Select an action for evaluation.

        :param batch_obs: (torch.Tensor) Tensor containing the observation.
        :return: (torch.Tensor) Selected action for evaluation.
        """  # noqa: E501

    @abc.abstractmethod
    def update(self):
        """
        Perform a training update.
        """  # noqa: E501
