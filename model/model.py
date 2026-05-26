

import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    

    def __init__(self, input_dim: int = 2560, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @classmethod
    def load(cls, checkpoint_path: str,
             map_location: str = 'cpu') -> 'ShallowMLP':
        """Load model from checkpoint saved as {'model_state_dict': ...,
        'input_dim': ...}."""
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        input_dim = ckpt.get('input_dim', 2560)
        model = cls(input_dim=input_dim)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        model.eval()
        return model
