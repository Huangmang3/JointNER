# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import dill
import parser
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from parser.utils import Config, Dataset, Embedding
from parser.utils.common import BOS, EOS, PAD, UNK
from parser.utils.field import ChartField, Field, RawField, SubwordField, NEField
from parser.utils.fn import download, get_rng_state, set_rng_state
from parser.utils.logging import init_logger, logger, get_logger, progress_bar
from parser.utils.metric import Metric, SpanMetric
from parser.utils.parallel import DistributedDataParallel as DDP
from parser.utils.parallel import is_master
from parser.utils.transform import Tree
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from parser.utils.alg import crf
from parser.model import CRFConstituencyModel

logger = get_logger(__name__)

class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              clip=5.0, epochs=5000, patience=100, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args).build(batch_size, buckets, True, dist.is_initialized())
        dev = Dataset(self.transform, args.dev).build(batch_size, buckets)
        test = Dataset(self.transform, args.test).build(batch_size, buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        else:
            from transformers import AdamW, get_linear_schedule_with_warmup
            steps = len(train.loader) * epochs // args.update_steps
            # steps = 600 * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        self.epoch, self.best_e, self.patience, self.best_metric, self.elapsed = 1, 1, patience, Metric(), timedelta()
        if self.args.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
            self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
            set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
            for k, v in self.checkpoint_state_dict.items():
                setattr(self, k, v)
            train.loader.batch_sampler.epoch = self.epoch

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self._train(train.loader)
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if dev_metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, dev_metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        parser = self.load(**args)
        loss, metric = parser._evaluate(test.loader)
        parser.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(batch_size, buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(batch_size, buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            # print(name, value)
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, reload=False, src='github', checkpoint=False, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            checkpoint (bool):
                If ``True``, loads all checkpoint states to restore the training process. Default: ``False``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from parser import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(parser.MODEL[src].get(path, path), reload=reload))
        cls = parser.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        parser = cls(args, model, transform)
        parser.checkpoint_state_dict = state['checkpoint_state_dict'] if args.checkpoint else None
        # parser.checkpoint_state_dict = None
        return parser

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def save_checkpoint(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        checkpoint_state_dict = {k: getattr(self, k) for k in ['epoch', 'best_e', 'patience', 'best_metric', 'elapsed']}
        checkpoint_state_dict.update({'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'rng_state': get_rng_state()})
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'checkpoint_state_dict': checkpoint_state_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

class CRFConstituencyParser(Parser):
    r"""
    The implementation of CRF Constituency Parser :cite:`zhang-etal-2020-fast`.
    """

    NAME = 'crf-constituency'
    MODEL = CRFConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART
        self.NE = self.transform.NE

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              mbr=True,
              delete={'TOP', 'S', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              #delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              verbose=True,
              **kwargs):

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, mbr=True,
                 delete={'TOP', 'S', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 #delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 verbose=True,
                 **kwargs):

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False, mbr=True, verbose=True, **kwargs):

        return super().predict(**Config().update(locals()))

    @classmethod
    def load(cls, path, reload=False, src='github', **kwargs):

        return super().load(path, reload, src, **kwargs)

    def _train(self, loader):
        self.model.train()
        bar = progress_bar(loader)
        # for i in self.CHART.vocab:
        #     print(i,self.CHART.vocab[i])
        # for i in self.NE.vocab:
        #     print(i,self.NE.vocab[i])
        # num_params = sum(p.numel() for p in self.model.parameters())
        # print('the number of model params: {}'.format(num_params))
        # exit()
        for i, batch in enumerate(bar, 1):
            words, *feats, trees, charts, nes= batch
            # import pdb
            # pdb.set_trace()
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_pos_span, s_ne_label = self.model(words, feats)
            loss, _ = self.model.loss(s_pos_span, s_ne_label, charts, nes, mask, self.args.mbr)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SpanMetric()

        for batch in loader:
            words, *feats, trees, charts, nes = batch
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
            s_pos_span, s_ne_label = self.model(words, feats)
            loss, s_pos_span = self.model.loss(s_pos_span, s_ne_label, charts, nes, mask, self.args.mbr)
            chart_preds, ne_preds = self.model.decode(s_pos_span, s_ne_label, mask, self.CHART.vocab, self.NE.vocab)
            final_labels=[]

            for tree, chart, ne in zip(trees, chart_preds, ne_preds):
                labels=[]
                for (i, j, pos_label), ne_label in zip(chart, ne):
                    if self.NE.vocab[ne_label]!="NON-NE":
                        label = self.NE.vocab[ne_label]+"-"+self.CHART.vocab[pos_label]
                    else:
                        label = self.CHART.vocab[pos_label]
                    labels.append((i,j,label))
                final_labels.append(labels)
            preds = [Tree.build(tree, [(i, j, label) for i, j, label in labels])
                     for tree, labels in zip(trees, final_labels)]
            total_loss += loss.item()
            metric([Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                   [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'trees': [], 'probs': [] if self.args.prob else None}
        for batch in progress_bar(loader):
            words, *feats, trees = batch
            word_mask = words.ne(self.args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)

            s_pos_span, s_ne_label = self.model(words, feats)
            if self.args.mbr:
                s_pos_span = crf(s_pos_span, mask, mbr=True)
            chart_preds, ne_preds = self.model.decode(s_pos_span, s_ne_label, mask, self.CHART.vocab, self.NE.vocab)
            for tree, chart, ne in zip(trees, chart_preds, ne_preds):
                labels=[]
                for (i, j, pos_label), ne_label in zip(chart, ne):
                    if self.NE.vocab[ne_label]!="NON-NE":
                        label = self.NE.vocab[ne_label]+"-"+self.CHART.vocab[pos_label]
                    else:
                        label = self.CHART.vocab[pos_label]
                    labels.append(label)
                preds['trees'].extend([Tree.build(tree, [(i, j, label) for (i, j, pos_label), label in zip(chart,labels)])])

        return preds

    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True)
        TAG, CHAR, ELMO, BERT = None, None, None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField('words',
                                pad=t.pad_token,
                                unk=t.unk_token,
                                bos=t.cls_token or t.cls_token,
                                eos=t.sep_token or t.sep_token,
                                fix_len=args.fix_len,
                                tokenize=t.tokenize,
                                fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS, eos=EOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=args.fix_len)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField('bert',
                                    pad=t.pad_token,
                                    unk=t.unk_token,
                                    bos=t.cls_token or t.cls_token,
                                    eos=t.sep_token or t.sep_token,
                                    fix_len=args.fix_len,
                                    tokenize=t.tokenize,
                                    fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
                BERT.vocab = t.get_vocab()
        TREE = RawField('trees')
        CHART = ChartField('charts')
        NE = NEField('nes')
        transform = Tree(WORD=(WORD, CHAR, ELMO, BERT), POS=TAG, TREE=TREE, CHART=CHART, NE=NE)

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        CHART.build(train)
        NE.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_pos_labels': len(CHART.vocab),
            'n_ne_labels': len(NE.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)