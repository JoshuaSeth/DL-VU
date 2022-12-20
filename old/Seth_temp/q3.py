import torch.nn as nn
import torch

class Elman(nn.Module):
    def __init__(self, input_size=300, output_size=300, hidden_size=300, activation=nn.ReLU):
        super().__init__()

        # Define two linear layers
        self.linear1 = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), activation())
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # Get the batch size, sequence length, and input size
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()

        # Initialize hidden with zeros if not given
        # hidden: (batch_size, input_size)
        if hidden is None:
            hidden = torch.zeros(batch_size, input_size, dtype=torch.float)
            hidden = hidden.to(DEVICE)

        # List to store the outputs at each time step
        outs = []

        # Iterate over the time steps in the input sequence
        for i in range(seq_len):
            # Concatenate the previous hidden state with the current input
            # prev_and_current: (batch_size, input_size + hidden_size)
            prev_and_current = torch.cat([x[:, i, :], hidden], dim=1)

            # Compute the hidden state for the current time step
            # hidden: (batch_size, hidden_size)
            hidden = self.linear1(prev_and_current)
            # Compute the output for the current time step
            # out: (batch_size, output_size)
            out = self.linear2(hidden)

            # Append the output at the current time step to the outputs list
            # out: (batch_size, 1, output_size)
            outs.append(out[:, None, :])
            
        # Save the final hidden state to be used as the initial hidden state
        # in the next forward pass
        self.hidden = hidden

        # Concatenate the outputs from each time step into a single tensor
        # outs: (batch_size, seq_len, output_size)
        outs.append(out[:, None, :])
            
        # Save the final hidden state to be used as the initial hidden state
        # in the next forward pass
        self.hidden = hidden

        return torch.cat(outs, dim=1), hidden