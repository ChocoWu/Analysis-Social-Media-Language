import os
import argparse
import logging

import torch
import torchtext
from torch.optim import Adam

import seq2seq
from seq2seq.trainer.supervised_trainer import SupervisedTrainer
from seq2seq.trainer.multi_task_train import MultiTaskTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder, Seq2seq
from seq2seq.models.multi_task import Multi_Task
from seq2seq.models.Classification import Classification
from seq2seq.loss.loss import Perplexity
from seq2seq.dataset.fields import SourceField, TargetField
from seq2seq.evaluator.multi_task_evaluator import Evaluator
from seq2seq.evaluator.predictor import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.tokenizer import tokenize
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data', default='./data/agr_en_train.csv')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data', default='./data/agr_en_dev.csv')
parser.add_argument('--test1_path', action='store', dest='test_path',
                    help='Path to test data', default='./data/test_1.csv')
parser.add_argument('--test2_path', action='store', dest='test_path',
                    help='the twitter test dataset', default='../data/test_2.csv')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment/',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='2019_11_21_18_10_05',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--num_class', type=int, default=3)

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Prepare dataset
norm_src = SourceField(tokenize=tokenize)
norm_tgt = TargetField(tokenize=tokenize)
class_src = SourceField(tokenize=tokenize)
class_tgt = SourceField(tokenize=tokenize)
max_len = 60  # the max length of a sentence

def norm_len_filter(example):
    return len(example.norm_src) <= max_len and len(example.norm_tgt) <= max_len

def class_len_filter(example):
    return len(example.class_src) <= max_len

train = None
dev = None
norm_input_vocab = None
norm_output_vocab = None
multi_task = None
optimizer = None
loss = None

if opt.load_checkpoint is not None:
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='csv',
        fields=[('class_src', class_src), ('class_tgt', class_tgt)],
        filter_pred=class_len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='csv',
        fields=[('class_src', class_src), ('class_tgt', class_tgt)],
        filter_pred=class_len_filter
    )
    # test1 = torchtext.data.TabularDataset(
    #     path=opt.test1_path, format='csv',
    #     fields=[('class_src', class_src), ('class_tgt', class_tgt)],
    #     filter_pred=class_len_filter
    # )
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    multi_task = checkpoint.model
    train.fields[seq2seq.class_src_field_name].vocab = checkpoint.input_vocab
    dev.fields[seq2seq.class_src_field_name].vocab = checkpoint.input_vocab
    # test1.fields[seq2seq.class_src_field_name].vocab = checkpoint.input_vocab
    # class_src.build_vocab(train)
    class_tgt.build_vocab(train)

    # weight = torch.ones(len(class_tgt.vocab))
    # pad = norm_tgt.vocab.stoi[norm_tgt.pad_token]
    # loss = Perplexity(weight, pad)
    # if torch.cuda.is_available():
    #     loss.cuda()

else:
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='csv',
        fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
        filter_pred=norm_len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='csv',
        fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
        filter_pred=norm_len_filter
    )
    # test = torchtext.data.TabularDataset(
    #     path=opt.test_path, format='csv',
    #     fields=[('norm_src', norm_src), ('norm_tgt', norm_tgt)],
    #     filter_pred=norm_len_filter
    # )
    norm_src.build_vocab(train, vectors='glove.twitter.27B.100d')
    norm_tgt.build_vocab(train, vectors='glove.twitter.27B.100d')

    norm_input_vocab = norm_src.vocab
    norm_output_vocab = norm_tgt.vocab

    weight = torch.ones(len(norm_tgt.vocab))
    pad = norm_tgt.vocab.stoi[norm_tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()


if multi_task is None:

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
        classification = Classification(hidden_size, opt.num_layer, opt.num_class, bidirectional=True, dropout_p=0.5,
                                        use_attention=False)
        multi_task = Multi_Task(encoder, decoder, classification)

if torch.cuda.is_available():
        multi_task.cuda()

# train
t = MultiTaskTrainer(loss=loss, batch_size=15,
                     checkpoint_every=100,
                     print_every=200, expt_dir=opt.expt_dir)

multi_task = t.train(multi_task, train,
                     num_epochs=16, dev_data=dev,
                     optimizer=optimizer,
                     teacher_forcing_ratio=0.5,
                     resume=opt.resume, pre_train=False)

# evaluator = Evaluator(loss=loss, batch_size=32)
# dev_loss, accuracy = evaluator.evaluate(multi_task, dev)
# print('dev_loss: {}, dev_accuracy: {}'.format(dev_loss, accuracy))
# assert dev_loss < 1.5

beam_search = Multi_Task(multi_task.encoder, TopKDecoder(multi_task.decoder, 3), multi_task.classification)

# if torch.cuda.is_available():
#     beam_search = beam_search.cuda()

predictor = Predictor(beam_search, norm_input_vocab, norm_output_vocab)
# inp_seq = ["This was largely accounted for by seed under 9 years old , about 90% of which is viable .",
#            "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"]
inp_seq = "MENTION MENTION weddings in the summer in Aruba ofc u guys r my bridesmaids"
seq = predictor.predict(inp_seq.split())
print(" ".join(seq[:-1]))
assert " ".join(seq[:-1]) == inp_seq[::-1]
