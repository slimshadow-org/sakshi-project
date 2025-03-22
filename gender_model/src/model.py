# src/model.py
import torch
import torch.nn as nn

class NameGenderClassifierCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=64, filter_sizes=[2, 3, 4], dropout=0.5):
        super(NameGenderClassifierCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 100)
        self.fc2 = nn.Linear(100, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding layer
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Transpose for convolution
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)
        
        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, seq_len - filter_size + 1)
            pool_out = torch.max_pool1d(conv_out, conv_out.shape[2])  # (batch_size, num_filters, 1)
            conv_outputs.append(pool_out.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate outputs from different filter sizes
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Fully connected layers with dropout
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x).squeeze()