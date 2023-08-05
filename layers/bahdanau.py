import torch
import torch.nn as nn
import math


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, attention_size: int, bias: bool = True):
        """
        Bahdanau Attention Mechanism.

        Args:
            encoder_hidden_size (int): The size of the encoder hidden states.
            decoder_hidden_size (int): The size of the decoder hidden states.
            attention_size (int): The size of the attention vector.
        """
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size

        self.W_enc = nn.Linear(encoder_hidden_size, attention_size)
        self.W_dec = nn.Linear(decoder_hidden_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

        if bias:
            self.bias = nn.Parameter(torch.empty(attention_size))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_dec.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, encoder_hidden_states: torch.Tensor, decoder_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass of the Bahdanau Attention Mechanism.

        Args:
            encoder_hidden_states (torch.Tensor): The encoder hidden states. Shape: (batch_size, seq_length, encoder_hidden_size)
            decoder_hidden_state (torch.Tensor): The current decoder hidden state. Shape: (batch_size, decoder_hidden_size)

        Returns:
            context_vector (torch.Tensor): The context vector for the current decoder step. Shape: (batch_size, encoder_hidden_size)
            attention_weights (torch.Tensor): The attention weights over the encoder hidden states. Shape: (batch_size, seq_length)
        """
        seq_length = encoder_hidden_states.size(1)

        # Calculate alignment scores
        encoder_score = self.W_enc(encoder_hidden_states)  # (batch_size, seq_length, attention_size)

        decoder_score = self.W_dec(decoder_hidden_state)  # (batch_size, 1, attention_size)

        # Combine encoder and decoder scores and apply the non-linearity
        b = self.bias if self.bias is not None else 0
        alignment_scores = torch.tanh(encoder_score + torch.unsqueeze(decoder_score, 1) + b)  # (batch_size, seq_length, attention_size)

        # Calculate attention weights
        attention_weights = self.V(alignment_scores).squeeze(-1)  # (batch_size, seq_length)

        # Apply softmax to get normalized attention weights
        attention_weights = torch.softmax(attention_weights, dim=-1)  # (batch_size, seq_length)

        # Calculate context vector as a weighted sum of encoder hidden states
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden_states.view(-1, seq_length, self.encoder_hidden_size))
        context_vector = context_vector.squeeze(1)  # (batch_size, encoder_hidden_size)

        return context_vector, attention_weights


class BahdanauEncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size: int, encoder_hidden_size: int, decoder_hidden_size: int, attention_size: int):
        """
        Encoder-Decoder LSTM with Bahdanau Attention.

        Args:
            input_size (int): The size of the input features.
            encoder_hidden_size (int): The size of the encoder hidden states.
            decoder_hidden_size (int): The size of the decoder hidden states.
            attention_size (int): The size of the attention vector.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size

        self.encoder_lstm = nn.LSTM(input_size, encoder_hidden_size, batch_first=True)
        self.attention = BahdanauAttention(encoder_hidden_size, decoder_hidden_size, attention_size)

        self.decoder_lstm = nn.LSTMCell(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
        self._attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Encoder-Decoder LSTM.

        Args:
            x (torch.Tensor): The input tensor. Shape: (n_batch, n_times, n_features)

        Returns:
            output (torch.Tensor): The output tensor. Shape: (n_batch, n_times, decoder_hidden_size)
        """
        n_batch, n_times, n_features = x.size()

        # Encoder LSTM
        encoder_hidden_states, (h_n_enc, c_n_enc) = self.encoder_lstm(x)

        # Decoder LSTM initial hidden state
        h_n_dec, c_n_dec = self.decoder_lstm(torch.zeros(n_batch, self.decoder_hidden_size + self.encoder_hidden_size).to(x.device))

        # Output tensor to store predictions
        output = torch.zeros(n_batch, n_times, self.decoder_hidden_size).to(x.device)
        attention_weights = list()

        # Iterate over each time step
        for t in range(n_times):

            context_vector, current_attention_weights = self.attention(encoder_hidden_states, h_n_dec)

            attention_weights.append(current_attention_weights)

            # Concatenate the context vector with the current decoder hidden state
            decoder_input = torch.cat((context_vector, h_n_dec), dim=-1)

            # Update the decoder LSTM hidden state at the current time step
            h_n_dec, c_n_dec = self.decoder_lstm(decoder_input, (h_n_dec, c_n_dec))

            # Compute output at the current time step
            output[:, t, :] = h_n_dec

        self._attention_weights = attention_weights
        return output
        # return encoder_hidden_states

    @property
    def attention_weights(self):
        return self._attention_weights
