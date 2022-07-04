import abc

# # Available actions
# ACTION_NOTHING = 0
# ACTION_ADD_SMALL_VM = 1
# ACTION_REMOVE_SMALL_VM = 2
# ACTION_ADD_MEDIUM_VM = 3
# ACTION_REMOVE_MEDIUM_VM = 4
# ACTION_ADD_LARGE_VM = 5
# ACTION_REMOVE_LARGE_VM = 6

class BaselineModel(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, S=None, M=None):
        raise NotImplementedError

class RandomModel(BaselineModel):
    def __init__(self, action_space):
        super().__init__(action_space)

    def __str__(self) -> str:
        return 'RandomModel'

    def predict(self, obs, S=None, M=None):
        return [self.action_space.sample()], (None, None, None)

class SimpleCPUThresholdModel(BaselineModel):
    def __init__(self, action_space):
        super().__init__(action_space)

    def __str__(self) -> str:
        return 'SimpleCPUThresholdModel'

    def predict(self, obs, S=None, M=None):
        # based on avgCPUUtilizationHistory
        avgCPUUtilizationHistory = obs[0][1]
        if avgCPUUtilizationHistory > 0.9:
            action = 5
        elif 0.7 < avgCPUUtilizationHistory < 0.9:
            action = 3
        elif 0.5 < avgCPUUtilizationHistory < 0.7:
            action = 1
        elif 0.3 < avgCPUUtilizationHistory < 0.5:
            action = 6
        elif 0.1 < avgCPUUtilizationHistory < 0.3:
            action = 4
        elif 0.0 < avgCPUUtilizationHistory < 0.1:
            action = 2
        else:
            action = 0
        return [action], (None, None, None)
