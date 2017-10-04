#!/usr/bin/env python
# 
# Thanks to Hao Cheng for this code

import os
import sys
import argparse
import glob

def tree2seq(args):
    files = glob.glob(os.path.join(args.indir, \
            '*.{}'.format(args.ext)))

    if not files:
        sys.stderr.write('Err: empty dir {}\n'.format(args.indir))
        sys.exit(1)
    if args.debug:
        if args.fn == -1:
            print files
        else:
            print files[args.fn]
        
    with open(args.outfile, 'w') as fout:
        for fn in files:
            with open(fn) as fin:
                new_tree = []
                for line in fin:
                    if not line.strip():
                        continue
                    if line[0] == '(':
                        # new tree
                        if new_tree != []:
                            fout.write('{}\n'.format(' '.join(new_tree)))
                            new_tree = []
                    new_tree.append(line.strip())
            if new_tree:
                fout.write('{}\n'.format(' '.join(new_tree)))


if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Convert Tree to Sequence')
    pa.add_argument('--indir', help='input directory')
    pa.add_argument('--ext', default='mrg', help='file extension')
    pa.add_argument('--outfile', help='output filename')
    pa.add_argument('--debug', action='store_true', dest='debug')
    pa.set_defaults(debug=False)
    pa.add_argument('--fn', type=int, default=-1)
    args = pa.parse_args()
    tree2seq(args)
    sys.exit(0)

