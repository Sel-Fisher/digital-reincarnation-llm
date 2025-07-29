# Project: Digital Reincarnation based on a Custom LLM

This project is a prototype of a service for creating a "digital twin" based on a person's communication style. It is implemented as a FastAPI microservice that allows you to upload a message history and fine-tune the `meta-llama/Meta-Llama-3-8B-Instruct` model using the LoRA technique.

## Project Structure

- `main.py`: The main file with FastAPI endpoints (`/upload-text`, `/train`).
- `parser.py`: A module for parsing and saving messages from a chat log.
- `train.py`: A module containing the logic for LoRA model fine-tuning.
- `requirements.txt`: A list of project dependencies.
- `examples/chat_example.json`: An example of an input dialog file.

## Requirements

- Python 3.10+
- `pip` and `venv`
- Git
- **NVIDIA GPU** with CUDA support (recommended, at least 16GB VRAM for the 8B model with quantization).
- Access to the Llama 3 model on Hugging Face.

## Setup and Launch

**1. Clone the repository:**
```bash
git clone <YOUR_REPOSITORY_URL>
cd digital-reincarnation-llm
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Log in to Hugging Face:**
You need an access token from Hugging Face to download the Llama 3 model.
```bash
huggingface-cli login
```
Paste your token when prompted. Make sure you have accepted the model's terms of use on its [Hugging Face page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

**5. Start the service:**
```bash
uvicorn main:app --reload
```
The service will be available at `http://127.0.0.1:8000`.

## How to Use the API

Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

### Step 1: Upload Data

- Go to the `POST /upload-text` endpoint.
- In the `speaker` field, enter `User1` or `User2`.
- In the `file` field, upload your `.json` dialog file (e.g., `examples/chat_example.json`).
- Click "Execute".

**Example using `curl`:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/upload-text' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'speaker=User1' \
  -F 'file=@./examples/chat_example.json'
```
A successful response means that the messages for `User1` have been filtered and saved to `./data/User1_messages.json`.

### Step 2: Start Training

- Go to the `POST /train` endpoint.
- In the request body, specify which speaker to train the model for:
  ```json
  {
    "speaker": "User1"
  }
  ```
- Click "Execute".

**Example using `curl`:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "speaker": "User1"
  }'
```
You will receive an immediate response indicating that training has started in the background. You can monitor the progress in the console where `uvicorn` is running.

### Step 3: Results

Once training is complete, the trained LoRA adapter will be saved in the `./output_lora/<speaker>` directory. For example, for `User1`, the path will be `./output_lora/User1`. This adapter can be used in the future to generate text in the style of the chosen person.