# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from parser.modules import (CharLSTM, ELMoEmbedding, IndependentDropout,
                           SharedDropout, TransformerEmbedding,
                           VariationalLSTM)
from parser.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from parser.modules import MLP, Biaffine
from parser.utils.alg import crf, cky_simple

class Model(nn.Module):
    
    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo_bos_eos=(True, True),
                 elmo_dropout=0.5,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=100,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # n_plm_embed=100
        # embed_dropout=0.2
        # n_lstm_hidden=300
        if encoder != 'bert':
            self.word_embed = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

            n_input = n_embed
            # n_input += n_pretrained
            if n_pretrained != n_embed:
                # n_input += n_pretrained
                n_input = n_pretrained
            if 'tag' in feat:
                self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                              embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'char' in feat:
                pass

                self.char_embed = CharLSTM(n_chars=n_chars,
                                           n_embed=n_char_embed,
                                           n_hidden=n_char_hidden,
                                           n_out=n_feat_embed,
                                           pad_index=char_pad_index,
                                           dropout=char_dropout)
                # n_input += n_feat_embed
               
            if 'lemma' in feat:
                self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                                embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'elmo' in feat:
                self.elmo_embed = ELMoEmbedding(n_out=n_plm_embed,
                                                bos_eos=elmo_bos_eos,
                                                dropout=elmo_dropout,
                                                finetune=finetune)
                n_input += self.elmo_embed.n_out
            if 'bert' in feat:
                self.bert_embed = TransformerEmbedding(model=bert,
                                                       n_layers=n_bert_layers,
                                                       n_out=n_plm_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=bert_pad_index,
                                                       mix_dropout=mix_dropout,
                                                       finetune=finetune)
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)
        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        else:
            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                mix_dropout=mix_dropout,
                                                finetune=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            word_embed = pretrained
            # if self.args.n_embed == self.args.n_pretrained:
            #     word_embed = pretrained
            #     # word_embed += pretrained
            # else:
            #     word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)
                # word_embed = torch.cat((word_embed, pretrained), -1)
        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            pass
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        
        if 'char' in self.args.feat:
            embed = self.embed_dropout(word_embed)[0]
        else:
            word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), -1)
        # word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # embed = torch.cat((word_embed, feat_embed), -1)
        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError

class CRFConstituencyModel(Model):
    r"""
    The implementation of CRF Constituency Parser :cite:`zhang-etal-2020-fast`,
    also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_span_mlp (int):
            Span MLP size. Default: 500.
        n_label_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_pos_labels,
                 n_ne_labels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, True),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=100,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_span_mlp=500,
                 n_label_mlp=100,
                 mlp_dropout=.33,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.pos_label_mlp_l = MLP(n_in=self.args.n_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.pos_label_mlp_r = MLP(n_in=self.args.n_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.ne_label_mlp_l = MLP(n_in=self.args.n_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.ne_label_mlp_r = MLP(n_in=self.args.n_hidden, n_out=n_label_mlp, dropout=mlp_dropout)

        self.pos_label_attn = Biaffine(n_in=n_span_mlp, n_out=n_pos_labels, bias_x=True, bias_y=True)
        self.ne_label_attn = Biaffine(n_in=n_label_mlp, n_out=n_ne_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """

        x = self.encode(words, feats)

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        pos_label_l = self.pos_label_mlp_l(x)
        pos_label_r = self.pos_label_mlp_r(x)
        ne_label_l = self.ne_label_mlp_l(x)
        ne_label_r = self.ne_label_mlp_r(x)
        s_pos_span = self.pos_label_attn(pos_label_l, pos_label_r).permute(0, 2, 3, 1)
        # import pdb;
        # pdb.set_trace()
        s_ne_label = self.ne_label_attn(ne_label_l, ne_label_r).permute(0, 2, 3, 1)
        # s_ne_label = self.ne_label_attn(ne_label_l, ne_label_r).unsqueeze(1).permute(0, 2, 3, 1)

        return s_pos_span, s_ne_label

    def loss(self, s_pos_span, s_ne_label, charts, nes, mask, mbr=False):
        import pdb
        pdb.set_trace()
        span_mask = charts.ge(0) & mask
        pos_span_loss, pos_span_probs = crf(s_pos_span,  mask, charts, mbr)
        ne_label_loss = self.criterion(s_ne_label[span_mask], nes[span_mask])
        loss = pos_span_loss + ne_label_loss 

        return loss, pos_span_probs

    def decode(self, s_pos_span, s_ne_label, mask, pos_vocab, ne_vocab):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all constituent labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """        
        chart_preds = cky_simple(s_pos_span, mask)
        ne_label_preds = s_ne_label.argmax(-1).tolist()
        ne_preds = []
        # import pdb
        # pdb.set_trace()
        for chart, ne_label in zip(chart_preds, ne_label_preds):
            ne_pred=[]
            tmpj=-1
            for i, j, label in chart:
                # ne_pred.append(ne_label[i][j])
                if pos_vocab.itos[label] == "PROPN":
                    ne_pred.append(ne_label[i][j])
                else:
                    ne_pred.append(ne_vocab.stoi["NON-NE"])
            ne_preds.append(ne_pred)
        return chart_preds, ne_preds