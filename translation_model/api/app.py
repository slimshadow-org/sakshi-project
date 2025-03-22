import os
from typing import List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import requests  # Import the requests library

app = FastAPI()

# Load API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("The GEMINI_API_KEY environment variable is not set.")

# Gemini API endpoint (using the v1beta endpoint for text generation)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key=" + GEMINI_API_KEY


class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = None


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str


# --- Helper Functions ---

def detect_language_and_options(text: str):
    """Detects the language and provides translation options using the requests library."""

    prompt = f"""Please identify the language of the text provided and then offer translation options as numbered choices (1-5).  Use this format:  "The text is in [Language].  Choose a language to translate to: 1. [Option 1], 2. [Option 2], 3. [Option 3], 4. [Option 4], 5. [Option 5]"
    
    Input Text: {text}"""  # Include the input text in the prompt

    request_data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(GEMINI_API_URL, json=request_data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()

        # Robust parsing, handling potential errors from the API
        try:
            # Accessing the response text correctly.  This is the most likely path.
            response_text = response_json['candidates'][0]['content']['parts'][0]['text']
            source_language = response_text.split("The text is in ")[1].split(".")[0].strip()
            options_str = response_text.split("Choose a language to translate to:")[1].strip()
            options_list = [opt.split(". ")[1].strip() for opt in options_str.split(", ")]
            while len(options_list) < 5:
                options_list.append("Option Not Available")
            options_list = options_list[:5]
            options = {str(i + 1): lang for i, lang in enumerate(options_list)}
            return source_language, options
        except (KeyError, IndexError, AttributeError) as e:  # Handle common parsing errors
            raise HTTPException(status_code=500, detail=f"Error parsing Gemini API response: {e}")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")
    except Exception as e: #Catch any other unexpected error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
def translate_with_gemini(text: str, source_language: str, target_language: str) -> str:
    """Translates text using the Gemini API via requests."""

    prompt = f"Translate the following text from {source_language} to {target_language}:\n\n{text}"

    request_data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(GEMINI_API_URL, json=request_data)
        response.raise_for_status()  # Important: Check for HTTP errors
        response_json = response.json()

        # Robustly extract the translated text, handling potential errors
        try:
            translated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return translated_text
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=500, detail=f"Error parsing Gemini API response: {e}")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred {e}")



@app.post("/translate", response_model=TranslationResponse, status_code=status.HTTP_200_OK)
async def translate(request: TranslationRequest):
    """Translates text from a source language to a target language."""

    if not request.text:
        raise HTTPException(status_code=400, detail="Text to translate cannot be empty.")
    if not request.target_language:
        raise HTTPException(status_code=400, detail="Target language must be provided.")

    if request.source_language:
        source_language = request.source_language
    else:
        try:
            source_language, _ = detect_language_and_options(request.text)
        except HTTPException as e:
            raise e

    supported_languages = ["English", "Hindi", "Telugu", "Marathi", "Bengali", "Tamil", "Spanish", "French", "German", "Japanese", "Chinese"]
    if request.target_language not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Target language '{request.target_language}' is not supported.")

    try:
        translated_text = translate_with_gemini(request.text, source_language, request.target_language)
        return TranslationResponse(translated_text=translated_text, source_language=source_language, target_language=request.target_language)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"An unexpected error occurred {e}")



@app.post("/detect_language", status_code=status.HTTP_200_OK)
async def detect_language(text: str = ""):
    """Detects the language of the input text and provides translation options."""
    if not text:
        raise HTTPException(status_code=400, detail="Text to detect cannot be empty.")
    try:
        source_language, options = detect_language_and_options(text)
        return {"source_language": source_language, "translation_options": options}
    except HTTPException as e:
        raise e
    except Exception as e:
         raise HTTPException(status_code=500,detail=f"An unexpected error occurred {e}")