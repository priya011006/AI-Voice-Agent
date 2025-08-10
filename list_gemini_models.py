import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Loads your .env, if you use one

api_key = os.getenv("GEMINI_API_KEY")  # Or paste your API key directly here as a string
genai.configure(api_key=api_key)

models = genai.list_models()
for model in models:
    print(model.name, model.supported_generation_methods)
    