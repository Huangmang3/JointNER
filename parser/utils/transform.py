# -*- coding: utf-8 -*-

import os
from collections.abc import Iterable

import sys  # 导入sys模块
sys.setrecursionlimit(10000)

import nltk
from parser.utils.logging import get_logger, progress_bar
from parser.utils.tokenizer import Tokenizer

logger = get_logger(__name__)


class Transform(object):
    r"""
    A Transform object corresponds to a specific data format.
    It holds several instances of data fields that provide instructions for preprocessing and numericalizing, etc.

    Attributes:
        training (bool):
            Sets the object in training mode.
            If ``False``, some data fields not required for predictions won't be returned.
            Default: ``True``.
    """

    fields = []

    def __init__(self):
        self.training = True

    def __len__(self):
        return len(self.fields)

    def __repr__(self):
        s = '\n' + '\n'.join([f" {f}" for f in self.flattened_fields]) + '\n'
        return f"{self.__class__.__name__}({s})"

    def __call__(self, sentences):
        # numericalize the fields of each sentence
        for sentence in progress_bar(sentences):
            # print(getattr(sentence, "charts"))
            # exit()
            for f in self.flattened_fields:
                # print(f.name)
                sentence.transformed[f.name] = f.transform([getattr(sentence, f.name)])[0]
        return self.flattened_fields

    def __getitem__(self, index):
        return getattr(self, self.fields[index])

    @property
    def flattened_fields(self):
        flattened = []
        for field in self:
            # print(field)
            if field not in self.src and field not in self.tgt:
                continue
            if not self.training and field in self.tgt:
                continue
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    flattened.append(f)
        # print(flattened)
        # exit()
        return flattened

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def append(self, field):
        self.fields.append(field.name)
        setattr(self, field.name, field)

    @property
    def src(self):
        raise AttributeError

    @property
    def tgt(self):
        raise AttributeError

    def save(self, path, sentences):
        with open(path, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences]) + '\n')


class Tree(Transform):
    r"""
    The Tree object factorize a constituency tree into four fields,
    each associated with one or more :class:`~supar.utils.field.Field` objects.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        CHART:
            The factorized sequence of binarized tree traversed in pre-order.
    """

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CHART', 'NE']

    def __init__(self, WORD=None, POS=None, TREE=None, CHART=None, NE=None):
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.CHART = CHART
        self.NE = NE

    @property
    def src(self):
        return self.WORD, self.POS, self.TREE

    @property
    def tgt(self):
        return self.CHART,  self.NE

    @classmethod
    def totree(cls, tokens, root='', special_tokens={'(': '-LRB-', ')': '-RRB-'}):
        r"""
        Converts a list of tokens to a :class:`nltk.tree.Tree`.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words or word/pos pairs.
            root (str):
                The root label of the tree. Default: ''.
            special_tokens (dict):
                A dict for normalizing some special tokens to avoid tree construction crash.
                Default: {'(': '-LRB-', ')': '-RRB-'}.

        Returns:
            A :class:`nltk.tree.Tree` object.

        Examples:
            >>> print(Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP'))
            (TOP ( (_ She)) ( (_ enjoys)) ( (_ playing)) ( (_ tennis)) ( (_ .)))
        """

        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        mapped = []
        for i, (word, pos) in enumerate(tokens):
            if word in special_tokens:
                tokens[i] = (special_tokens[word], pos)
                mapped.append((i, word))
        tree = nltk.Tree.fromstring(f"({root} {' '.join([f'( ({pos} {word}))' for word, pos in tokens])})")
        for i, word in mapped:
            tree[i][0][0] = word
        return tree

    @classmethod
    def binarize(cls, tree):
        r"""
        Conducts binarization over the tree.

        First, the tree is transformed to satisfy `Chomsky Normal Form (CNF)`_.
        Here we call :meth:`~nltk.tree.Tree.chomsky_normal_form` to conduct left-binarization.
        Second, all unary productions in the tree are collapsed.

        Args:
            tree (nltk.tree.Tree):
                The tree to be binarized.

        Returns:
            The binarized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> print(Tree.binarize(tree))
            (TOP
              (S
                (S|<>
                  (NP (_ She))
                  (VP
                    (VP|<> (_ enjoys))
                    (S::VP (VP|<> (_ playing)) (NP (_ tennis)))))
                (S|<> (_ .))))

        .. _Chomsky Normal Form (CNF):
            https://en.wikipedia.org/wiki/Chomsky_normal_form
        """

        tree = tree.copy(True)
        if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree(f"{tree.label()}|<>", [tree[0]])
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend([child for child in node])
                if len(node) > 1:
                    for i, child in enumerate(node):
                        if not isinstance(child[0], nltk.Tree):
                            node[i] = nltk.Tree(f"{node.label()}|<>", [child])
        tree.chomsky_normal_form('left', 0, 0)
        tree.collapse_unary(joinChar='::')

        return tree

    @classmethod
    def factorize(cls, tree, delete_labels=None, equal_labels=None):
        r"""
        Factorizes the tree into a sequence.
        The tree is traversed in pre-order.

        Args:
            tree (nltk.tree.Tree):
                The tree to be factorized.
            delete_labels (set[str]):
                A set of labels to be ignored. This is used for evaluation.
                If it is a pre-terminal label, delete the word along with the brackets.
                If it is a non-terminal label, just delete the brackets (don't delete children).
                In `EVALB`_, the default set is:
                {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
                Default: ``None``.
            equal_labels (dict[str, str]):
                The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
                The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
                Default: ``None``.

        Returns:
            The sequence of the factorized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> Tree.factorize(tree)
            [(0, 5, 'TOP'), (0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]
            >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
            [(0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]

        .. _EVALB:
            https://nlp.cs.nyu.edu/evalb/
        """

        def track(tree, i):
            label = tree.label()
            if delete_labels is not None and label in delete_labels:
                label = None
            if equal_labels is not None:
                label = equal_labels.get(label, label)
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return (i+1 if label is not None else i), []
            j, spans = i, []
            for child in tree:
                j, s = track(child, j)
                spans += s
            if label is not None and j > i:
                spans = [(i, j, label)] + spans
            return j, spans
        return track(tree, 0)[1]

    @classmethod
    def build(cls, tree, sequence):
        r"""
        Builds a constituency tree from the sequence. The sequence is generated in pre-order.
        During building the tree, the sequence is de-binarized to the original format (i.e.,
        the suffixes ``|<>`` are ignored, the collapsed labels are recovered).

        Args:
            tree (nltk.tree.Tree):
                An empty tree that provides a base for building a result tree.
            sequence (list[tuple]):
                A list of tuples used for generating a tree.
                Each tuple consits of the indices of left/right boundaries and label of the constituent.

        Returns:
            A result constituency tree.

        Examples:
            >>> tree = Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP')
            >>> sequence = [(0, 5, 'S'), (0, 4, 'S|<>'), (0, 1, 'NP'), (1, 4, 'VP'), (1, 2, 'VP|<>'),
                            (2, 4, 'S::VP'), (2, 3, 'VP|<>'), (3, 4, 'NP'), (4, 5, 'S|<>')]
            >>> print(Tree.build(tree, sequence))
            (TOP
              (S
                (NP (_ She))
                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                (_ .)))
        """

        root = tree.label()
        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

        def track(node):
            i, j, label = next(node)
            # print(i, j, label)
            if j == i+1:
                children = [leaves[i]]
            else:
                children = track(node) + track(node)
            if label is None or label.endswith('|<>'):
                return children
            labels = label.split('::')
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]
        return nltk.Tree(root, track(iter(sequence)))

    def load(self, data, lang=None, max_len=None, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`TreeSentence` instances.
        """
        max_len=510
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                trees = [nltk.Tree.fromstring(s) for s in f]
            self.root = trees[0].label()
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            trees = [self.totree(i, self.root) for i in data]

        i, sentences = 0, []
        for tree in progress_bar(trees):
            sentences.append(TreeSentence(self, tree))
            i += 1
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]
        return sentences


class Batch(object):

    def __init__(self, sentences):
        self.sentences = sentences
        self.transformed = {f.name: f.compose([s.transformed[f.name] for s in sentences])
                            for f in sentences[0].transform.flattened_fields}
        self.fields = list(self.transformed.keys())

    def __repr__(self):
        s = ', '.join([f"{name}" for name in self.fields])
        return f"{self.__class__.__name__}({s})"

    def __getitem__(self, index):
        return self.transformed[self.fields[index]]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self.transformed:
            return self.transformed[name]
        if hasattr(self.sentences[0], name):
            return [getattr(s, name) for s in self.sentences]
        raise AttributeError


class Sentence(object):

    def __init__(self, transform):
        self.transform = transform

        # mapping from each nested field to their proper position
        self.maps = dict()
        # names of each field
        self.keys = set()
        for i, field in enumerate(self.transform):
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    self.maps[f.name] = i
                    self.keys.add(f.name)
        # original values and numericalized values of each position
        self.values = []
        self.transformed = {key: None for key in self.keys}

    def __contains__(self, key):
        return key in self.keys

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.maps:
            return self.values[self.maps[name]]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if 'keys' in self.__dict__ and name in self:
            index = self.maps[name]
            if index >= len(self.values):
                self.__dict__[name] = value
            else:
                self.values[index] = value
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)


class TreeSentence(Sentence):
    r"""
    Args:
        transform (Tree):
            A :class:`Tree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
    """

    def __init__(self, transform, tree):
        super().__init__(transform)
        words, tags = zip(*tree.pos())
        chart = [[None]*(len(words)+1) for _ in range(len(words)+1)]
        ne = [["NON-NE"]*(len(words)+1) for _ in range(len(words)+1)]
        for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
            if "-PROPN" in label and "|<>" not in label:
                if "S::" in label:
                    ne[i][j]=label[3:label.index("-")]
                else:
                    ne[i][j]=label[:label.index("-")]
                label="PROPN"
            elif "-PROPN" in label and "|<>" in label:
                label=label[label.index("-")+1:]
            # if label.count(":")>10:
            chart[i][j] = label 
            # ne[i][j]=label 

        self.values = [words, tags, tree, chart, ne]

    def __repr__(self):
        return self.values[2].pformat(1000000)

    def __len__(self):
        count=0
        for words in self.values[0]:
            count+=len(words)
        return count

    def pretty_print(self):
        self.values[2].pretty_print()
