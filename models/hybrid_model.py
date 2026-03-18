import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                 num_lstm_layers: int = 1, num_transformer_layers: int = 2, 
                 dropout: float = 0.1):
        """
        Hybrid LSTM-Transformer architecture for time series forecasting.
        LSTM captures local sequential patterns.
        Transformer captures global context.
        """
        super(HybridLSTMTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Local context: LSTM
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, 
                            num_layers=num_lstm_layers, batch_first=True, 
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        
        # Global context: Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)
        
        # Regression output
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Predict single scalar (e.g., next step log return)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. Project to d_model space
        x = self.input_projection(x)
        
        # 2. LSTM for sequential features
        x, _ = self.lstm(x)
        
        # 3. Positional Encoding + Transformer for global dependency
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 4. Use the output of the last time step for prediction
        x_last = x[:, -1, :]
        
        # 5. Output layer
        out = self.fc_out(x_last)
        return out

if __name__ == "__main__":
    # Smoke test
    model = HybridLSTMTransformer(input_dim=5)
    dummy_input = torch.randn(32, 30, 5) # batch_size, seq_len, features
    output = model(dummy_input)
    print("Model Output Shape:", output.shape) # Expected: (32, 1)

