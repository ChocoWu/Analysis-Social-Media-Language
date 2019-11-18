#!/user/bin/env python3
# -*- utf-8 -*-
# author shengqiong.wu

from __future__ import print_function, division

import torch
import torchtext
import seq2seq
from seq2seq.loss.loss import NLLLoss
import torch
import torch.nn as nn


class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.norm_src),
            device=device, repeat=False)

        class_batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.class_src),
            device=device, repeat=False,
        )
        tgt_vocab = data.fields[seq2seq.norm_tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.norm_tgt_field_name].pad_token]
        pred_result = []
        gold_result = []
        with torch.no_grad():
            for batch, class_batch in zip(batch_iterator, class_batch_iterator):
                input_variables, input_lengths = getattr(batch, seq2seq.norm_src_field_name)
                target_variables = getattr(batch, seq2seq.norm_tgt_field_name)
                class_input, class_lengths = getattr(class_batch, seq2seq.class_src_field_name)
                class_output, _ = getattr(class_batch, seq2seq.class_tgt_field_name)

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()
                    class_input = class_input.cuda()
                    class_output = class_output.cuda()

                (decoder_outputs, decoder_hidden, other), class_result = model(input_variables, input_lengths,
                                                                               target_variables,
                                                                               class_input, class_output, class_lengths)

                pred_result.extend(torch.max(class_result[0], dim=1)[1].cpu().numpy().tolist())
                gold_result.extend(class_output.squeeze().cpu().numpy().tolist())

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()


        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, pred_result, gold_result

    def pre_train_evaluator(self, model, data):
        """ Evaluate a model on pre_train given dataset and return performance.
        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.norm_src),
            device=device, repeat=False)

        tgt_vocab = data.fields[seq2seq.norm_tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.norm_tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, seq2seq.norm_src_field_name)
                target_variables = getattr(batch, seq2seq.norm_tgt_field_name)

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                (decoder_outputs, decoder_hidden, other) = model(input_variables, input_lengths,
                                                                 target_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()


        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
