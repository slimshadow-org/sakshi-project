from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, conlist
import torch
import torch.nn as nn
import os  # Import the 'os' module
from typing import List

# --- Model Definition (same as before) ---
class NameGenderClassifierCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=64, filter_sizes=[2, 3, 4], dropout=0.5):
        super(NameGenderClassifierCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 100)
        self.fc2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pool_out = torch.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pool_out.squeeze(2))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()



# --- Utility Function (same as before, but adapted) ---

def tokenize_name(name, char_to_idx, max_length):
    """Tokenizes and pads a name."""
    name = str(name).lower()
    tokens = [char_to_idx.get(char, char_to_idx.get(' ', 1)) for char in name]

    # Pad or truncate
    if len(tokens) < max_length:
        tokens = tokens + [char_to_idx['<PAD>']] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return tokens


# --- FastAPI Setup ---

app = FastAPI(title="Indian Name Gender Prediction API",
              description="Predicts the gender of Indian names using a CNN model.",
              version="1.0")

# --- Model Loading (on startup) ---

MODEL_PATH = "models/indian_name_gender_model.pt"  # Correct path within the space


def load_model():
    """Loads the model, char_to_idx, and max_name_length."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        char_to_idx = checkpoint['char_to_idx']
        max_name_length = checkpoint['max_name_length']
        config = checkpoint['model_config']

        model = NameGenderClassifierCNN(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_filters=config['num_filters'],
            filter_sizes=config['filter_sizes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model, char_to_idx, max_name_length, device
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

# Load model at startup
try:
    model, char_to_idx, max_name_length, device = load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    raise  # Re-raise the exception to halt startup

# --- Pydantic Models (for request/response validation) ---

class PredictionRequest(BaseModel):
    names: conlist(str, min_length=1) = Field(..., example=["Aarav", "Anika"])
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability threshold for classifying as male.")

class PredictionResponse(BaseModel):
    predictions: List[dict] = Field(..., example=[
        {"name": "Aarav", "predicted_gender": "Male", "male_probability": 0.95, "confidence": 0.95},
        {"name": "Anika", "predicted_gender": "Female", "male_probability": 0.05, "confidence": 0.95}
    ])


# --- Prediction Function ---

def predict_gender(name: str, model, char_to_idx, max_length, device, threshold: float = 0.5) -> tuple[str, float, float]:
    """Predicts gender for a single name.  Includes threshold."""
    tokenized_name = tokenize_name(name, char_to_idx, max_length)
    input_tensor = torch.tensor([tokenized_name], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        predicted_gender = 'Male' if probability >= threshold else 'Female'
        confidence = probability if probability >= threshold else 1 - probability
    return predicted_gender, probability, confidence

# --- API Endpoints ---

@app.get("/", response_model=str)
async def read_root():
	return "Welcome to the Indian Name Gender Prediction API.  Use the /predict endpoint."

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predicts the gender of one or more Indian names."""
    try:
        predictions = []
        for name in request.names:
            gender, prob, conf = predict_gender(name, model, char_to_idx, max_name_length, device, request.threshold)
            predictions.append({
                "name": name,
                "predicted_gender": gender,
                "male_probability": prob,
                "confidence": conf
            })
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predict_single")
async def predict_single(name: str = Query(..., description="The name to predict."),
                         threshold: float = Query(0.5, ge=0.0, le=1.0, description="Probability threshold for classifying as male.")):
    """Predicts gender for a *single* name, provided as a query parameter."""
    try:
        gender, prob, conf = predict_gender(name, model, char_to_idx, max_name_length, device, threshold)
        return {
            "name": name,
            "predicted_gender": gender,
            "male_probability": prob,
            "confidence": conf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))