import os
import json
import time
import requests
import re
from openai import OpenAI
from cloud_optimizer.config import settings

def get_action(client: OpenAI, obs: dict, model_name: str):
    traffic = obs['current_traffic']
    active = obs['active_instances']
    cpu = obs['cpu_utilization']
    
    prompt = f"""
    You are a Cloud DevOps AI. 
    Current Traffic: {traffic}
    Active Servers: {active}
    CPU Utilization: {cpu}
    
    Output ONLY valid JSON with "action_type" ("SCALE_UP", "SCALE_DOWN", "NO_OP") and "instance_count".
    Example: {{"action_type": "SCALE_UP", "instance_count": 2}}
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_text = completion.choices[0].message.content
        match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
        
    return {"action_type": "NO_OP", "instance_count": 0}

def run_inference():
    # Use environment variables per checklist
    api_base = os.getenv("API_BASE_URL", settings.API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", settings.MODEL_NAME)
    hf_token = os.getenv("HF_TOKEN", settings.HF_TOKEN)
    env_url = settings.ENV_URL

    client = OpenAI(base_url=api_base, api_key=hf_token)

    # 1. Fetch Tasks
    try:
        tasks_res = requests.get(f"{env_url}/tasks", timeout=5).json()
        tasks = tasks_res.get("tasks", ["easy", "medium", "hard"])
    except Exception:
        tasks = ["easy", "medium", "hard"]

    for task_id in tasks:
        # [START] log
        start_payload = {
            "env_id": "cloud-cost-optimizer",
            "task_id": task_id,
            "model_name": model_name
        }
        print(f"[START] {json.dumps(start_payload)}")

        # Reset Env
        try:
            requests.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=5)
        except Exception as e:
            print(f"Error resetting env: {e}")
            continue
        
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            # Get current state
            try:
                state_res = requests.get(f"{env_url}/state", timeout=5).json()
                obs = state_res["observation"]
                
                # Predict action
                action = get_action(client, obs, model_name)
                
                # Step in env
                step_res = requests.post(f"{env_url}/step", json=action, timeout=5).json()
                
                done = step_res["done"]
                reward = step_res["reward"]
                
                # [STEP] log
                step_log = {
                    "step": step_count,
                    "observation": obs,
                    "action": action,
                    "reward": reward["value"],
                    "done": done
                }
                print(f"[STEP] {json.dumps(step_log)}")
            except Exception as e:
                print(f"Error during step {step_count}: {e}")
                break

        # [END] log
        try:
            score_res = requests.get(f"{env_url}/grader", timeout=5).json()
            score = score_res["score"]
        except:
            score = 0.0
            
        end_log = {
            "task_id": task_id,
            "score": score
        }
        print(f"[END] {json.dumps(end_log)}")

if __name__ == "__main__":
    # Ensure server is up if running locally
    # In the submission environment, the server is usually started via Docker
    run_inference()
