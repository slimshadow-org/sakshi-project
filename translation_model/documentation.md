FastAPI Translation API with Google Gemini

This document describes a FastAPI application that provides translation services using Google Gemini's generative capabilities. The API supports both translation between languages and language detection.

**The application is hosted at: https://sameernotes-translation-prediction-space.hf.space**

Installation (for local development/deployment)

1.  Install Dependencies:

    pip install fastapi uvicorn pydantic requests

2.  Set Gemini API Key:

    Obtain an API key for Google Gemini from the Google Cloud Console.  Set it as an environment variable:

    export GEMINI_API_KEY="your_gemini_api_key_here"

Running the Application (locally)

Use uvicorn to start the FastAPI server:

uvicorn main:app --reload

Replace `main` with the name of your Python file if it's different. The `--reload` flag enables automatic reloading of the server when you make code changes.

API Endpoints

The API provides two main endpoints:

1. /translate (POST)

Translates text from a source language to a target language.

Request Body:

{
  "text": "string",        // The text to be translated. (Required)
  "target_language": "string", // The target language (e.g., "English", "Hindi"). (Required)
  "source_language": "string"  // Optional: The source language.  If not provided, it will be detected.
}

Response (Success - 200 OK):

{
  "translated_text": "string",  // The translated text.
  "source_language": "string",  // The detected or provided source language.
  "target_language": "string"   // The target language.
}

Error Responses:

*   400 Bad Request:
    *   {"detail": "Text to translate cannot be empty."} (if text is missing)
    *   {"detail": "Target language must be provided."} (if target_language is missing)
    *   {"detail": "Target language '...' is not supported. Supported languages: ..."} (if the target language is not in the supported list)
*   500 Internal Server Error:
    *   {"detail": "Error communicating with Gemini API: ..."} (if there's a network issue)
    *   {"detail": "Error parsing Gemini API response: ..."} (if the API response is in an unexpected format)
    *   {"detail": "Language detection failed: ..."} (if language detection fails)
    *   {"detail": "Translation failed: ..."} (if the translation fails)
    *   {"detail": "An unexpected error occurred: ..."} (if there's any other unexpected error)

Example (using curl):

curl -X 'POST' \
  'https://sameernotes-translation-prediction-space.hf.space/translate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Hello, world!",
  "target_language": "Spanish"
}'

2. /detect_language (POST)

Detects the language of the input text and provides a list of possible translation options.

Request Body:

{
  "text": "string"  // The text to detect the language of. (Required)
}

Response (Success - 200 OK):

{
  "source_language": "string",  // The detected language.
  "translation_options": {     // A dictionary of translation options (number: language).
    "1": "string",
    "2": "string",
    "3": "string",
    "4": "string",
    "5": "string"
  }
}
Error Responses:
* 400 Bad Request:
   * {"detail": "Text to detect cannot be empty."}
* 500 Internal Server Error:
   * {"detail": "Error communicating with Gemini API: ..."}
    *   {"detail": "Error parsing Gemini API response: ..."}
    *  {"detail": "An unexpected error occurred: ..."}