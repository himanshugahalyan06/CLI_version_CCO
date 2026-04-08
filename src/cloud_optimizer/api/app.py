from scalar_fastapi import get_scalar_api_reference
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cloud_optimizer.core.schemas import Action, Observation
from cloud_optimizer.core.environment import CloudOptimizerEnv
from cloud_optimizer.config import settings
from fastapi.responses import RedirectResponse
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

app = FastAPI(title="Cloud Cost Optimizer API")
env = CloudOptimizerEnv()

class ResetRequest(BaseModel):
    task_id: str

@app.get("/", include_in_schema=False)
async def redirect_to_ui():
    return RedirectResponse(url="/scalar")

@app.get("/tasks")
async def get_tasks():
    logger.info("Fetching available tasks")
    return {
        "tasks": list(env.tasks.keys()),
        "action_schema": Action.model_json_schema()
    }

@app.post("/reset")
async def reset(req: ResetRequest = None): 
    if req is None:
        req = ResetRequest(task_id="easy")
    
    logger.info(f"Resetting environment for task: {req.task_id}")
    try:
        env.reset(req.task_id)
    except Exception as e:
        logger.error(f"Error resetting environment: {e}")
        env.reset()
        
    return {"message": "Reset successful", "task_id": env.current_task_id}

@app.post("/step")
async def step_environment(action: Action):
    if env.current_step >= len(env.traffic_profile):
        logger.warning("Attempted to step in a finished episode")
        raise HTTPException(status_code=400, detail="Episode already done. Please /reset.")
    
    logger.debug(f"Step {env.current_step}: Action {action.action_type} with count {action.instance_count}")
    response = env.step(action)
    return response.model_dump()

@app.get("/state")
async def get_state():
    return {"observation": env._get_observation().model_dump()}

@app.get("/grader")
async def get_grader():
    score = env.get_grader_score()
    logger.info(f"Grader requested. Current score: {score}")
    return {"score": score}

@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )

def main():
    import uvicorn
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

if __name__ == "__main__":
    main()
