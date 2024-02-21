# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parser.utils import Config
from parser.utils.logging import init_logger, logger
from parser.utils.parallel import get_free_port
from parser.parser import CRFConstituencyParser

def init(parser):
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--conf', '-c', default='', help='path to config file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device_count = torch.cuda.device_count()
    if device_count > 1:
        os.environ['MASTER_ADDR'] = 'tcp://localhost'
        os.environ['MASTER_PORT'] = get_free_port()
        mp.spawn(parse, args=(args,), nprocs=device_count)
    else:
        parse(0 if torch.cuda.is_available() else -1, args)


def parse(local_rank, args):
    Parser = args.pop('Parser')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_logger(logger, f"{args.path}.{args.mode}.log", 'a' if args.get('checkpoint') else 'w')
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',
                                init_method=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                world_size=torch.cuda.device_count(),
                                rank=local_rank)
    torch.cuda.set_device(local_rank)
    logger.info('\n' + str(args))

    args.local_rank = local_rank
    if args.mode == 'train':
        parser = Parser.load(**args) if args.checkpoint else Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(**args)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(**args)
        parser.predict(**args)

def main():
    parser = argparse.ArgumentParser(description='Create CRF Constituency Parser.')
    parser.set_defaults(Parser=CRFConstituencyParser)
    parser.add_argument('--mbr', action='store_true', help='whether to use MBR decoding')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', help='path to train file')
    subparser.add_argument('--dev', help='path to dev file')
    subparser.add_argument('--test', help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    # subparser.add_argument('--bert', default='bert-base-chinese', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', help='path to dataset')
    subparser.add_argument('--pred', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()