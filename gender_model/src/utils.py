# src/utils.py

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