import abc


class AbstractAgent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def init(self):
        """Initialize the agent."""
        pass

    @abc.abstractmethod
    def observe(
        self,
        batch_obs,
        batch_actions,
        batch_rewards,
        batch_next_obs,
        batch_dones,
    ):
        """Store a transition in the replay buffer."""
        pass

    @abc.abstractmethod
    def act_train(self, batch_obs):
        """Select an action for training."""
        pass

    @abc.abstractmethod
    def act_eval(self, obs):
        """Select an action for evaluation."""
        pass

    @abc.abstractmethod
    def update(self):
        """Perform a training update."""
        pass
