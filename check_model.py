import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

# Configure genai
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
print("Available models and their supported methods:")
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"Supported Methods: {model.supported_generation_methods}")
    print("-" * 50)