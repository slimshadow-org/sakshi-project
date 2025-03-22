# src/predict.py
import torch
import argparse
from model import NameGenderClassifierCNN  # Import from model.py
from utils import tokenize_name  # Import from utils.py
import os

def predict_gender(name, model, char_to_idx, max_length, device, threshold=0.5):
    """Predicts the gender of a single name."""
    model.eval()
    tokenized_name = tokenize_name(name, char_to_idx, max_length)
    input_tensor = torch.tensor([tokenized_name], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        predicted_gender = 'Male' if probability >= threshold else 'Female'
        confidence = probability if probability >= threshold else 1 - probability

    return predicted_gender, probability, confidence

def load_model(model_path, device):
    """Loads the model and necessary components."""
    checkpoint = torch.load(model_path, map_location=device)
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
    return model, char_to_idx, max_name_length

def main():
    parser = argparse.ArgumentParser(description="Predict gender from Indian names.")
    parser.add_argument("--model_path", type=str, default="/home/sameer/sakshi/gender_model/models/indian_name_gender_model.pt",
                        help="Path to the saved model file.")
    parser.add_argument("--names", type=str, nargs='+',
                        help="Names to predict (space-separated).  Or use --input_file.")
    parser.add_argument("--input_file", type=str,
                        help="Path to a text file containing names (one name per line).")
    parser.add_argument("--output_file", type=str,
                        help="Path to a file to write the predictions (optional).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for classifying as male (default: 0.5).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda, default: cpu).")
    args = parser.parse_args()


    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Determine the device to use
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available.  Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device.lower())

    # Load the model
    model, char_to_idx, max_name_length = load_model(args.model_path, device)

    # Get the names to predict
    if args.names:
        names = args.names
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found at {args.input_file}")
            return
        with open(args.input_file, 'r') as f:
            names = [line.strip() for line in f]
    else:
        print("Error: Please provide names using --names or --input_file.")
        return

    # Make predictions
    predictions = []
    for name in names:
        gender, male_prob, confidence = predict_gender(name, model, char_to_idx, max_name_length, device, args.threshold)
        predictions.append((name, gender, male_prob, confidence))

    # Print or save the results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write("Name,Predicted_Gender,Male_Probability,Confidence\n")
            for name, gender, male_prob, confidence in predictions:
                f.write(f"{name},{gender},{male_prob:.4f},{confidence:.4f}\n")
        print(f"Predictions saved to {args.output_file}")
    else:
        print("Name\t\tPrediction\tMale Prob\tConfidence")
        print("-" * 60)
        for name, gender, male_prob, confidence in predictions:
            print(f"{name:<12}\t{gender:<10}\t{male_prob:.4f}\t\t{confidence:.4f}")

if __name__ == "__main__":
    main()