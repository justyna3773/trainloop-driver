import numpy as np
import pprint

class Utils:
    FEATURE_NAMES = ["vmAllocatedRatio",
                    "avgCPUUtilization",
                    "p90CPUUtilization",
                    "avgMemoryUtilization",
                    "p90MemoryUtilization",
                    "waitingJobsRatioGlobal",
                    # "waitingJobsRatioRecent"
                    ]

    ACTION_NAMES = {
        0:'NOTHING',
        1:'ADD_SMALL_VM',
        2:'REMOVE_SMALL_VM',
        3:'ADD_MEDIUM_VM',
        4:'REMOVE_MEDIUM_VM',
        5:'ADD_LARGE_VM',
        6:'REMOVE_LARGE_VM'
        }

def get_action_observation_map(predictions):
    predictions = np.array(predictions)
    action_observation_map = {action:np.argwhere(predictions == i) for i, action in enumerate(Utils.ACTION_NAMES.values())}
    action_observation_map_count = {k:v.shape[0] for k, v in action_observation_map.items()}
    print('Observation count for each action:')
    pprint.pprint(action_observation_map_count)
    return action_observation_map

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
