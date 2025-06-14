import os
from dotenv import load_dotenv
from openai import OpenAI
import openai

# Enable verbose HTTP logging (optional)
openai.debug = True

# Load environment variables from a .env file, if present
load_dotenv()

# Instantiate a reusable OpenAI client
client = OpenAI(
    base_url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),
) 