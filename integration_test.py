import os
import argparse
import logging

import torch
import torchtext
from torch.optim import Adam

# import seq2seq
from seq2seq.trainer.supervised_trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder, Seq2seq
from seq2seq.loss.loss import Perplexity
from seq2seq.dataset.fields import SourceField, TargetField
from seq2seq.evaluator.evaluator import Evaluator
from seq2seq.evaluator.predictor import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.tokenizer import tokenize


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data', default='./data/train.csv')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data', default='./data/dev.csv')
parser.add_argument('--test_path', action='store', dest='test_path',
                    help='Path to test data', default='./data/test.csv')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment/',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--lr', type=float, default=0.001)

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Prepare dataset
norm_src = SourceField(tokenize=tokenize)
norm_tgt = TargetField(tokenize=tokenize)
max_len = 50  # the max length of a sentence


def len_filter(example):
    return len(example.norm_src) <= max_len and len(example.norm_tgt) <= max_len


train = torchtext.data.TabularDataset(
    path=opt.train_path, format='csv',
    fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path=opt.dev_path, format='csv',
    fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
    filter_pred=len_filter
)
test = torchtext.data.TabularDataset(
    path=opt.test_path, format='csv',
    fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
    filter_pred=len_filter
)
norm_src.build_vocab(train, max_size=200000, vectors='glove.twitter.27B.100d')
norm_tgt.build_vocab(train, max_size=200000, vectors='glove.twitter.27B.100d')
input_vocab = norm_src.vocab
output_vocab = norm_tgt.vocab

# Prepare loss
weight = torch.ones(len(norm_tgt.vocab))
pad = norm_tgt.vocab.stoi[norm_tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 100
        bidirectional = True
        encoder = EncoderRNN(len(norm_src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional,
                             rnn_cell='lstm',
                             variable_lengths=True)
        decoder = DecoderRNN(len(norm_tgt.vocab), max_len, hidden_size * 2,
                             dropout_p=0.2, use_attention=True,
                             bidirectional=bidirectional,
                             rnn_cell='lstm',
                             eos_id=norm_tgt.eos_id, sos_id=norm_tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=40,
                          checkpoint_every=100,
                          print_every=200, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=5, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

evaluator = Evaluator(loss=loss, batch_size=32)
dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
print('dev_loss: {}, dev_accuracy: {}'.format(dev_loss, accuracy))
# assert dev_loss < 1.5

beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 3))

# if torch.cuda.is_available():
#     beam_search = beam_search.cuda()

predictor = Predictor(beam_search, input_vocab, output_vocab)
# inp_seq = ["This was largely accounted for by seed under 9 years old , about 90% of which is viable .",
#            "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"]
inp_seq = "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"
seq = predictor.predict(inp_seq.split())
print(" ".join(seq[:-1]))
assert " ".join(seq[:-1]) == inp_seq[::-1]
