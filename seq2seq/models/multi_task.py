import torch
import torch.nn as nn
import torch.nn.functional as  F


class Multi_Task(nn.Module):
    """ sequence-to-sequence architecture with configurable encoder decoder and classification.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        classification (Classification): object of Classification
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: norm_input, norm_lengths, norm_target, teacher_forcing_ratio, class_input, class_y
        - **norm_input** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **class_input** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **norm_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **norm_target** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **class_y** (list, optional): list of sequences, whose length is the batch size and within which
          each label is a list of token IDs. This information is forwarded to the classification network.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: norm_result(decoder_outputs, decoder_hidden, ret_dict), class_result(classification, attn)
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
        - **classification** (batch, num_class): batch-length list of tensors with size(num_class) containing the
          outputs of the classification networks
        - **attn**: (batch) optional, if use attention in classification network, we will get the attention weights of each word
          in one sentence,

    """
    def __init__(self, encoder, decoder, classification, decode_function=F.log_softmax):
        super(Multi_Task, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classification = classification
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, norm_input, norm_lengths=None, norm_target=None, class_input=None,
                class_y=None, class_lengths=None, teacher_forcing_ratio=0):
        norm_encoder_outputs, norm_encoder_hidden = self.encoder(norm_input, norm_lengths)
        class_encoder_outputs, class_encoder_hidden = self.encoder(class_input, class_lengths)
        norm_result = self.decoder(inputs=norm_target,
                                   encoder_hidden=norm_encoder_hidden,
                                   encoder_outputs=norm_encoder_outputs,
                                   function=self.decode_function,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        class_result = self.classification(class_encoder_outputs)

        return norm_result, class_result
