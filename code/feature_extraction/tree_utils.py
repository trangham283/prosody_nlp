#!/usr/bin/env python
# Tree utils for completing brackets and filling out missing words 

import os
import sys
import argparse
import random
import re

def merge_dels(token_list):
    new_list = []
    for i, s in enumerate(token_list):
        current_s = s
        prev_s = token_list[i-1] if i>0 else None
        if prev_s == "TO_DELETE" and current_s == "TO_DELETE": continue
        else: new_list.append(current_s)
    return new_list

def detach_brackets(toks):
    new_list = []
    for tok in toks:
        num_close = tok.count(')')
        if num_close <= 1:
            new_list.append(tok)
        else:
            new_tok = ' '.join([')']*num_close)
            new_list.append(new_tok)
    return new_list


def add_brackets(toks):
    line = ' '.join(toks)
    num_open = line.count('(')
    num_close = line.count(')')
    if num_open == num_close:
        if '(' in toks[0] and ')' in toks[-1]:
            full_sent = toks[:]
            valid = 1
        else:
            print line
            valid = 0
            full_sent = ['('] + toks[:] + [')']
    else:
        valid = 0
        if num_open < num_close:
            add_open = num_close - num_open
            extra_open = ['(']*add_open
            full_sent = extra_open + toks
        else:
            add_close = num_open - num_close
            extra_close = [')']*add_close
            full_sent = toks + extra_close
    return full_sent, valid

def match_length(parse, sent):
    line = ' '.join(parse)
    PUNC = ['.', ',', ':', '``', '\'\'', ';', '?', '!', '$', '"', '%', '*', '&']
    tree = []
    sent_toks = sent[:]
    dec_toks = parse[:]
    num_toks = len(sent_toks)
    num_parse = line.count('XX') 
    num_puncs = sum([line.count(x) for x in PUNC])
    num_out = num_puncs + num_parse
    if num_toks == num_out:
        new_tree = dec_toks[:]
    else:
        if num_out < num_toks: # add 'XX' in this case
            num_X = num_toks - num_out  
            for _ in range(num_X):
                if len(dec_toks) > 3:
                    x_add = random.choice(range(len(dec_toks) - 2)) 
                    # offset a bit so never insert at very beginning or very end
                    dec_toks.insert(x_add + 2, 'XX')
                else:
                    dec_toks.insert(1, 'XX')
            new_tree = dec_toks[:]
        else: # remove XXs 
            num_X = num_out - num_toks
            x_indices = [i for i, x in enumerate(dec_toks) if x == "XX"]
            if num_X < len(x_indices):
                x_remove = random.sample(set(x_indices), num_X)
                for k in x_remove:
                    dec_toks[k] = "TO_DELETE"
                for _ in range(len(x_remove)):
                    dec_toks.remove("TO_DELETE")
            # else: do nothing
            new_tree = dec_toks[:]
    return new_tree

def delete_constituents(parse):
    PUNC = ['.', ',', ':', '``', '\'\'', ';', '?', '!', '$', '"', '%', '*', '&']
    RM_CONSTITUENTS = ['(-NONE-', '(.', '(-DFL-', '(,']
    new_tree = parse[:]
    for i in range(len(new_tree)-1):
        this_tok = new_tree[i]
        next_tok = new_tree[i+1]
        if this_tok[0] == '(' and next_tok[0] in PUNC:
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"
        if this_tok[0] == '(' and next_tok[0] == ')':
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"
        if this_tok in RM_CONSTITUENTS:
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"

    num_del = new_tree.count("TO_DELETE")
    for _ in range(num_del): 
        new_tree.remove("TO_DELETE")
    if num_del == 0:
        return new_tree
    else:
        return delete_empty_constituents(new_tree)

def delete_empty_constituents(parse):
    new_tree = parse[:]
    for i in range(len(new_tree)-1):
        this_tok = new_tree[i]
        next_tok = new_tree[i+1]
        if this_tok[0] == '(' and next_tok[0] == ')':
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"

    num_del = new_tree.count("TO_DELETE")
    for _ in range(num_del): 
        new_tree.remove("TO_DELETE")
    if num_del == 0:
        return new_tree
    else:
        return delete_empty_constituents(new_tree)

# old version of delete_empty_constituents
# don't use
def delete_empty_constituents_2(parse):
    new_tree = parse[:]
    for i in range(len(new_tree)-1):
        this_tok = new_tree[i]
        next_tok = new_tree[i+1]
        if this_tok[0] == '(' and next_tok[0] == ')':
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"

    # There are a few cases of nested empty constituents,
    # take care of such cases:
    # merge consecutive "TO_DELETE" tokens
    tok_tmp = merge_dels(new_tree)
    del_constituents = [i for i, x in enumerate(tok_tmp) if 
            x == "TO_DELETE" and (i+1) < len(tok_tmp) and tok_tmp[i+1][0] == ")" and 
            tok_tmp[i-1][0] =="("]
    while len(del_constituents) > 0:
        for idx in del_constituents:
            if tok_tmp[idx+1] == ")" or tok_tmp[idx+1][:2] == ")_":
                tok_tmp[idx-1:idx+2] = ["TO_DELETE"]*3
            else: 
                # this is to take care of the difference between single ')'
                # or things like '))))'
                tok_tmp[idx-1:idx+1] = ["TO_DELETE"]*2
                tok_tmp[idx+1] = tok_tmp[idx+1][1:]
            tok_tmp = merge_dels(tok_tmp)
            del_constituents = [i for i, x in enumerate(tok_tmp)
                    if x == "TO_DELETE" and (i+1) < len(tok_tmp) and tok_tmp[i+1][0] == ")" and
                    tok_tmp[i-1][0] == "("]
    
    num_del = tok_tmp.count("TO_DELETE")
    for _ in range(num_del): 
        tok_tmp.remove("TO_DELETE")
    return tok_tmp


def merge_sent_tree(parse, sent):
    tree = []
    word_idx = 0
    for token in parse:
        tok = token
        if token == 'XX': 
            if word_idx < len(sent):
                tok = '(XX {})'.format(sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        elif token[0] == ')':
            tok = ')'
        elif token[0] != '(':
            if word_idx < len(sent):
                tok = '({} {})'.format(token, sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        tree.append(tok)
    new_tree = []
    idx = 0
    k = 0
    while idx < len(tree):
        token = tree[idx]
        if token == ')':
            k = 1
            while (idx + k) < len(tree):
                if tree[idx+k] != ')':
                    break
                k += 1
            token = ')' * k
            idx += k - 1
        idx += 1
        new_tree.append(token)

    return new_tree

def linearize_tree(line, rm_func_tag=False, pos_norm=True, dec_bracket=False,\
        lower=True):
    keep_set = ['-NONE-', ',', ':', '``', '\'\'', '.']

    #items = line.strip().split()
    items = line[:]
    sent = []
    tree = []
    tag_stack = []
    for idx, token in enumerate(items):
        token = token.strip()
        if token[0] == '(':
            # a tree part
            next_token = items[idx + 1]
            if next_token[0] == '(':
                # not POS
                # push the tag in stack
                if len(token) > 1:
                    if token[1] == '(':
                        tag_stack.append('(')
                        tree.append('(')
                    tok = token.strip('(')
                    if rm_func_tag:
                        try:
                            tok = tok.replace('-', ' ').replace('=', ' ').strip().split()[0]
                        except:
                            sys.stderr.write('''Err: rm-func-tag {} token
                                    {}\n'''.format(tok, token))
                            sys.exit(1)
                    tag_stack.append(tok)
                    tree.append('({}'.format(tok))
                else:
                    tag_stack.append(token)
                    tree.append(token)
            else:
                # current is POS
                if token[1:] == next_token[:-1] or\
                        token[1:] in keep_set:
                    tree.append(token.strip('('))
                elif pos_norm:
                    tree.append('XX')
                else:
                    tree.append(token.strip('('))
        elif token[0] == ')':
            # bracket annotation
            for i in range(len(token)):
                try:
                    tag = tag_stack.pop()
                except:
                    sys.stderr.write('Err: bracket does not match!\n')
                    sys.stderr.write('''current partial tree
                            {}\n'''.format(' '.join(tree)))
                    sys.stderr.write('current token {}\n'.format(token))
                    sys.stderr.write('current tree {}\n'.format(line))
                    sys.exit(1)
                if dec_bracket:
                    tree.append(')_{}'.format(tag))
                else:
                    tree.append(')')
        else:
            # word
            if lower:
                sent.append(token.strip(')').lower())
            else:
                sent.append(token.strip(')'))

    return sent, tree


