import pytest
from cloud_optimizer.core.environment import CloudOptimizerEnv
from cloud_optimizer.core.schemas import Action, ActionType

def test_env_initialization():
    env = CloudOptimizerEnv()
    obs = env.reset("easy")
    assert obs.active_instances == 2
    assert env.current_step == 0
    assert env.current_task_id == "easy"

def test_env_step_scale_up():
    env = CloudOptimizerEnv()
    env.reset("easy")
    initial_instances = env.active_instances
    
    action = Action(action_type=ActionType.SCALE_UP, instance_count=3)
    response = env.step(action)
    
    assert env.active_instances == initial_instances + 3
    assert response.observation.active_instances == initial_instances + 3
    assert env.current_step == 1

def test_env_step_scale_down():
    env = CloudOptimizerEnv()
    env.reset("easy")
    initial_instances = env.active_instances
    
    action = Action(action_type=ActionType.SCALE_DOWN, instance_count=1)
    response = env.step(action)
    
    assert env.active_instances == initial_instances - 1
    assert env.active_instances >= 0

def test_env_done():
    env = CloudOptimizerEnv()
    env.reset("easy")
    task_length = len(env.traffic_profile)
    
    for i in range(task_length):
        response = env.step(Action(action_type=ActionType.NO_OP, instance_count=0))
        if i < task_length - 1:
            assert not response.done
        else:
            assert response.done

def test_grader_score():
    env = CloudOptimizerEnv()
    env.reset("easy")
    # Simulate a perfect run (simplified logic check)
    score = env.get_grader_score()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
