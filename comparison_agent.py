from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from uagents import Model
from uagents.query import query
import google.generativeai as genai
import uvicorn

AGENT_ADDRESS = "agent1qfk78dqd7djfe30wayk4xvearaqcu08pgtsefhn0rn9tlt8ke3mv6a5j7s8"


class TestRequest(Model):
    message: str


class AgentRequest(Model):
    image: str
    prompt: str


async def agent_query(req):
    response = await query(destination=AGENT_ADDRESS, message=req, timeout=15.0)
    data = json.loads(response.decode_payload())
    return data["text"]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
genai.configure(api_key='AIzaSyANvgl8Qc8lW3bPOxtcFzj7yFgkjbPBxZE')
comparison_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')


class ComparisonRequest(BaseModel):
    description: str
    task: str


@app.post("/compare-task")
async def compare_task(request: ComparisonRequest):
    try:
        # Generate comparison
        prompt = f"""I will give you a detailed explanation of an image , and a task that the user had to perform. you have 
        to compare the image description and the task and then you have to return how much you are confident that the task 
        was completed. the format of the output should be mandatorily in pure json with 'result' being the key and either 'true' 
        if you think task is completed or  'false' if you think task is not completed. here is the image description {request.description} and here 
        is the task user was asked to do {request.task}"""
        response = comparison_model.generate_content([prompt])

        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return "Hello from the Agent controller"


@app.post("/endpoint")
async def make_agent_call(req: AgentRequest):
    try:

        request_data = json.dumps({"image": req.image, "prompt": req.prompt})
        res = await agent_query(request_data)
        return {"message": "successful call - agent response", "response": res}
    except Exception as e:
        return {"message": "unsuccessful agent call", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("comparison_agent:app", host="0.0.0.0", port=8000, reload=True)