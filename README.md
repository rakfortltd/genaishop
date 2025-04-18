# genaishop
Gen AI Dummy Applications for Learning &amp; Testing

# Installation 
cd genaishop

python3.11 -m venv venvrag

source venvrag/bin/activate

pip install - r requirements.txt

# Peform HuggingFace login
huggingface-cli login

Enter Hugging API token

# To run chainlit user interface
chainlit run app.py --port 8005

# To run API service for testing
uvicorn fastapi_app:app --port 8080
