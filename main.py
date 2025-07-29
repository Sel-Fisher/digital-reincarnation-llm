import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal

import parser
import train

app = FastAPI(
    title="Digital Reincarnation LLM",
    description="A service to fine-tune a Llama 3 model to mimic a user's communication style.",
    version="0.1.0",
)


class TrainRequest(BaseModel):
    speaker: Literal['User1', 'User2'] = Field(
        ...,
        description="The speaker for whom the model will be trained."
    )


@app.get("/", summary="Root Endpoint")
def read_root():
    return {"message": "Welcome to the Digital Reincarnation API"}


@app.post("/upload-text", summary="Upload Chat History")
async def upload_text(
        speaker: Literal['User1', 'User2'] = Form(...),
        file: UploadFile = File(...)
):
    """
    Get JSON file from Chat History and speaker name.
    Save speaker messages for future use.
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .json file.")

    try:
        contents = await file.read()
        dialog_data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in the uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    try:
        saved_path = parser.save_speaker_messages(dialog_data, speaker)
        return {
            "message": f"Successfully processed chat for speaker '{speaker}'",
            "saved_path": saved_path
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/train", summary="Start LoRA Fine-tuning")
def start_training(
        request: TrainRequest,
        background_tasks: BackgroundTasks
):
    """
    Run additional learning to model for a speaker on background.
    """
    speaker = request.speaker

    background_tasks.add_task(train.run_training, speaker)

    return {
        "message": f"Training process started in the background for speaker '{speaker}'. "
                   f"Check server logs for progress. Results will be saved in ./output_lora/{speaker}"
    }