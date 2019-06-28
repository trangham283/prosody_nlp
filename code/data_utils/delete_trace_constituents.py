#!/usr/bin/env python
# 

import os
import sys
import argparse
import re

def merge_dels(token_list):
    new_list = []
    for i, s in enumerate(token_list):
        current_s = s
        prev_s = token_list[i-1] if i>0 else None
        if prev_s == "TO_DELETE" and current_s == "TO_DELETE": continue
        else: new_list.append(current_s)
    return new_list

def delete_trace(args):
    with open(args.infile) as fin,  open(args.outfile, 'w') as fout:
        for line in fin:
            #line_tmp = line.strip().replace('(', ' ( ').replace(')', ' ) ')
            toks = line.strip().split()
            if "(-NONE-" not in toks:
                new_tree = toks[:]
                fout.write('{}\n'.format(' '.join(new_tree)))
            else:
                tok_tmp = toks[:]
                none_indices = [i for i, x in enumerate(toks) if x == "(-NONE-"]
                for idx in none_indices:
                    tok_tmp[idx:idx+2] = ["TO_DELETE"]*2

                # merge consecutive "TO_DELETE" tokens
                tok_tmp = merge_dels(tok_tmp)
                del_constituents = [i for i, x in enumerate(tok_tmp) if \
                        x == "TO_DELETE" and tok_tmp[i+1][0] == ")" and \
                        tok_tmp[i-1][0] =="("]                
                         
                while len(del_constituents) > 0:
                    for idx in del_constituents:
                        if tok_tmp[idx+1] == ")":
                            tok_tmp[idx-1:idx+2] = ["TO_DELETE"]*3
                        else:
                            tok_tmp[idx-1:idx+1] = ["TO_DELETE"]*2
                            tok_tmp[idx+1] = tok_tmp[idx+1][1:]
                    tok_tmp = merge_dels(tok_tmp)
                    del_constituents = [i for i, x in enumerate(tok_tmp) \
                            if x == "TO_DELETE" and tok_tmp[i+1][0] == ")" and \
                            tok_tmp[i-1][0] == "("]
                
                num_del = tok_tmp.count("TO_DELETE")
                for _ in range(num_del): tok_tmp.remove("TO_DELETE")
                new_tree = tok_tmp[:]
                fout.write('{}\n'.format(' '.join(new_tree)))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Delete traces and parent constituents.')
    pa.add_argument('--infile', help='input tree filename')
    pa.add_argument('--outfile', help='output filename')
    args = pa.parse_args()
    delete_trace(args)
    sys.exit(0)

