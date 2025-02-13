import torch
import torch.nn as nn
from .customlayers import CustomLinear

class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = CustomLinear(input_dim, hidden_dim)
        self.linear2 = CustomLinear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class MoeLayers(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_experts, top_k=2):
        super(MoeLayers, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.k = top_k

        self.experts = nn.ModuleList([FeedForward(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = CustomLinear(self.input_dim, self.num_experts, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x_squashed = x.view(-1, self.input_dim)  # (batch_size * seq_len, input_dim)

        # Compute the gate logits
        gate_logits = self.gate(x_squashed)  # (batch_size * seq_len, num_experts)
        gate_logits = torch.softmax(gate_logits, dim=-1)
        weights, selected_experts = torch.topk(gate_logits, self.k, dim=-1)  # (batch_size * seq_len, k)

        weights = nn.functional.softmax(weights, dim=1, dtype=x.dtype)  # (batch_size * seq_len, k)

        results = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(x_squashed[batch_idx])

        return results.view_as(x)

if __name__ == "__main__":
    model = MoeLayers(10, 20, 4, 2)
    x = torch.randn(1, 4, 10)
    out = model(x)
    print(out.shape)



