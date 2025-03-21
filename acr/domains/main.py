#!/usr/bin/env python
# coding=utf-8

from .loop_inv import add_loop_inv_args, LoopInvDomain
from .arc import ARCDomain
from .apps import add_apps_args, APPSDomain, APPSDomain_multi
import argparse
def add_domain_args(parser):
    add_loop_inv_args(parser)
    add_apps_args(parser)
    parser.add_argument('--domain', type=str, default='loop_inv',)
    parser.add_argument("--multiprocess", action=argparse.BooleanOptionalAction,default=False, help="multiprocess")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction,default=True, help="use cache")
    parser.add_argument("--use-sandbox", action=argparse.BooleanOptionalAction,default=False, help="use sandbox")

def get_domain(args):
    if args.domain == 'loop_inv':
        return LoopInvDomain(args)
    elif args.domain == 'arc':
        return ARCDomain(args)
    elif args.domain == 'apps':
        return APPSDomain(args)
    elif args.domain == 'apps_multi':
        return APPSDomain_multi(args)
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))

