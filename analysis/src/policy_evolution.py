from typing import Optional
from analysis.src.drl_agent_interpreter import DRLAgentInterpreter

class PolicyEvolution:
    def __init__(self,
                 n_training_steps: int,
                 step_value: int,
                 drl_agent_type: str,
                 dnn_model_architecture: str,
                 base_path: str,
                 observations_path: str,
                 n_samples: int = 500,
                 ) -> None:
        self.n_training_steps = n_training_steps
        self.step_value = step_value
        self.drl_agent_type = drl_agent_type
        self.dnn_model_architecture = dnn_model_architecture
        self.base_path = base_path
        self.observations_path = observations_path
        self.n_samples = n_samples

        self.training_steps_save_intervals = self.__get_training_steps_save_intervals()
        self.drl_agent_interpreters = self.__get_drl_agent_interpreters()


    def __get_training_steps_save_intervals(self):
        n_steps = self.n_training_steps // self.step_value
        return [1] + [(self.step_value*n)+self.step_value for n in range(n_steps)]

    def __get_drl_agent_interpreters(self):
        agent_paths = [f'{self.base_path}_{i}_{n}' for i, n in enumerate(self.training_steps_save_intervals, start=0)]

        return [DRLAgentInterpreter(drl_agent_type=self.drl_agent_type,
                                    dnn_model_architecture=self.dnn_model_architecture,
                                    agent_path=agent_path,
                                    observations_path=self.observations_path,
                                    n_samples=self.n_samples) for agent_path in agent_paths]
    
    def plot_mean_attributions(self, absolute_values: bool):
        for drl_agent_interpreter, i in zip(self.drl_agent_interpreters, self.training_steps_save_intervals):
            drl_agent_interpreter.plot_mean_attributions(absolute_values=absolute_values, title_postfix=f' - after {i} training steps')

    def explain_example(self, idx: int):
        for drl_agent_interpreter, i in zip(self.drl_agent_interpreters, self.training_steps_save_intervals):
            drl_agent_interpreter.feature_attributor.explain_example(idx=idx, title_postfix=f' - after {i} training steps')
