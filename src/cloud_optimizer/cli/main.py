import typer
import requests
import json
import re
import time
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from openai import OpenAI
from loguru import logger
import sys

from cloud_optimizer.config import settings
from cloud_optimizer.api.app import main as run_server

app = typer.Typer(help="Cloud Cost Optimizer - Industry Ready CLI")
console = Console()

# Configure Loguru for CLI
logger.remove()
logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

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
            return json.loads(match.group(0)), None
    except Exception as e:
        return {"action_type": "NO_OP", "instance_count": 0}, str(e)
        
    return {"action_type": "NO_OP", "instance_count": 0}, "Failed to parse JSON"

@app.command()
def serve():
    """Start the Cloud Cost Optimizer API server."""
    console.print(Panel("[bold green]Starting Cloud Cost Optimizer API Server...[/bold green]"))
    run_server()

@app.command()
def simulate(
    task: str = typer.Option("easy", help="Task difficulty: easy, medium, hard"),
    model: Optional[str] = typer.Option(None, help="LLM model name to use"),
    api_key: Optional[str] = typer.Option(None, help="OAI-compatible API Key"),
    api_base: Optional[str] = typer.Option(None, help="OAI-compatible API Base URL")
):
    """Run an AI simulation against the environment."""
    
    model_name = model or settings.MODEL_NAME
    key = api_key or settings.HF_TOKEN
    base_url = api_base or settings.API_BASE_URL
    
    console.print(Panel(
        f"[bold blue]Starting Simulation[/bold blue]\n"
        f"[cyan]Task:[/cyan] {task}\n"
        f"[cyan]Model:[/cyan] {model_name}\n"
        f"[cyan]Base URL:[/cyan] {base_url}"
    ))
    
    # Initialize OpenAI client with validator-injected or CLI-provided credentials
    client = OpenAI(base_url=base_url, api_key=key)
    
    # Check if internal server is up
    server_awake = False
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Waiting for internal server...", total=None)
        for _ in range(10): 
            try:
                requests.post(f"{settings.ENV_URL}/reset", json={"task_id": task}, timeout=2)
                server_awake = True
                break
            except Exception:
                time.sleep(1)
    
    if not server_awake:
        console.print("[bold red]Error:[/bold red] Could not connect to the API server. Please run [bold]cco serve[/bold] first.")
        raise typer.Exit(1)

    done = False
    step_count = 0
    total_reward = 0.0
    
    table = Table(title=f"Simulation Progress - Task: {task}")
    table.add_column("Step", justify="right")
    table.add_column("Traffic", justify="right")
    table.add_column("Servers", justify="right")
    table.add_column("CPU", justify="right")
    table.add_column("Action", style="magenta")
    table.add_column("Reward", justify="right")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        sim_task = progress.add_task(description="Simulating steps...", total=None)
        
        while not done:
            step_count += 1
            try:
                state_res = requests.get(f"{settings.ENV_URL}/state", timeout=5).json()
                obs = state_res["observation"]
                
                action_payload, error = get_action(client, obs, model_name)
                
                step_res = requests.post(f"{settings.ENV_URL}/step", json=action_payload, timeout=5).json()
                
                done = step_res["done"]
                reward = float(step_res['reward']['value'])
                total_reward += reward
                
                table.add_row(
                    str(step_count),
                    str(obs['current_traffic']),
                    str(obs['active_instances']),
                    f"{obs['cpu_utilization']:.2f}",
                    f"{action_payload['action_type']}({action_payload['instance_count']})",
                    f"{reward:+.2f}"
                )
                
                if error:
                    console.print(f"[yellow]Warning Step {step_count}: {error}[/yellow]")
                
            except Exception as e:
                console.print(f"[bold red]Critical Error at step {step_count}: {e}[/bold red]")
                break
    
    console.print(table)
    
    # Final score
    try:
        score_res = requests.get(f"{settings.ENV_URL}/grader", timeout=5).json()
        score = score_res['score']
    except:
        score = 0.0
        
    color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
    console.print(Panel(f"[bold {color}]Simulation Complete![/bold {color}]\n[bold]Final Score:[/bold] {score:.3f}\n[bold]Total Steps:[/bold] {step_count}"))

if __name__ == "__main__":
    app()
