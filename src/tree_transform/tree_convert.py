#!/usr/bin/env python
# TODO: describe
# Process PTB tree and MS-State hypothesis tree

from parse_analyzer import render_tree, init, treebanks, parse_errors
from parse_analyzer import head_finder, tree_transform
from parse_analyzer import pstree, transform_search
import sys, argparse, os
from parse_analyzer.classify_english import classify
from collections import defaultdict
from StringIO import StringIO
from difflib import SequenceMatcher
from nltk.tree import Tree
from nltk.draw.tree import TreeView
import pandas as pd 


# file with limited human annotation
ann_df = pd.read_csv('../human-ann.tsv', sep='\t')
ann_sents = set(ann_df.sent_id)

# replace word by word 
def replace_words(ptb_tree, ms_words, i1, i2, j1, j2):
    for k, sidx in enumerate(range(i1, i2)):
        this_node = ptb_tree.get_nodes(start=sidx+k, end=sidx+k+1)
        leaf = this_node[-1]
        assert leaf.is_terminal()
        leaf.word = ms_words[j1+k]
    ptb_tree.check_consistency()

# delete one word at a time
def delete_word(ptb_tree, i1):
    to_remove = ptb_tree.get_nodes(start=i1, end=i1+1)
    for node in to_remove:
        tree_transform.remove_node_by_node(node, True)
    # update spans
    for node in ptb_tree.get_nodes():
        if node.span[0] > i1: 
            node.span = (node.span[0]-1, node.span[1])
        if node.span[1] > i1: 
            node.span = (node.span[0], node.span[1]-1)
    ptb_tree.check_consistency()

def delete_words(ptb_tree, i1, i2):
    for k in range(i1, i2)[::-1]:
        # delete from back to front so it's easier to update spans
        delete_word(ptb_tree, k)

def insert_words(ptb_tree, ms_words, i1, i2, j1, j2):
    to_insert = ms_words[j1:j2]
    str_mrg = ['(XX ' + w +')' for w in to_insert]
    mrg = '(X ' + ' '.join(str_mrg) + ' )'
    node_to_add = pstree.tree_from_text(mrg)
    # adjust span of new node
    for n in node_to_add.get_nodes():
        n.span = (n.span[0] + i1, n.span[1] + i1)
        
    # adjust span of original tree
    offset = j2 - j1
    for node in ptb_tree.get_nodes():
        if node.span[0] >= i1: node.span = (node.span[0] + offset, node.span[1])
        if node.span[1] > i1: node.span = (node.span[0], node.span[1] + offset)

    # choose left sibling's parent by default
    left_sibling = ptb_tree.get_nodes(end=node_to_add.span[0], \
            request='highest')
    right_sibling = ptb_tree.get_nodes(start=node_to_add.span[1], \
            request='highest')
   
    if left_sibling.parent != right_sibling.parent:
        print "Warning: left and right parent don't match in insertion step"
    
    sibling = left_sibling
    new_parent = sibling.parent
    node_to_add.parent = new_parent
    idx = new_parent.subtrees.index(sibling)
    node_to_add.parent.subtrees.insert(idx+1, node_to_add)
    ptb_tree.check_consistency()

def merge_trees(t1, t2):
    left = treebanks.homogenise_tree(t1)
    right = treebanks.homogenise_tree(t2)
    sl, el = left.span
    sr, er = right.span
    # take the highest node under ROOT, usually "S" that has same span
    right_sub = right.get_nodes(start=sr, end=er)[1]
    for n in right_sub.get_nodes():
        n.span = (n.span[0] + el, n.span[1] + el)
    left.subtrees.append(right_sub)
    right_sub.parent = left
    left.span = (left.span[0], left.span[1] + er)
    return left 

def split_tree(tree, idx):
    t1 = pstree.tree_from_text(str(tree))
    t2 = pstree.tree_from_text(str(tree))
    delete_words(t2, 0, idx)
    delete_words(t1, idx, tree.span[1])
    return t1, t2

def convert_tree(ptb_tree, ms_tree, rev):
    treebanks.remove_function_tags(ptb_tree)
    treebanks.remove_function_tags(ms_tree)
    ms_words = ms_tree.word_yield(as_list=True)
    ptb_words = ptb_tree.word_yield(as_list=True)
    ptb_words = [x.lower() for x in ptb_words]

    #print ptb_words
    #print ms_words
    #print

    sseq = SequenceMatcher(None, ptb_words, ms_words)
    # .get_opcodes returns ops to turn a into b 
    ops = []
    for info in sseq.get_opcodes():
        if info[0] != 'equal': ops.append(info)

    for op in ops[::-1]:
        tag, i1, i2, j1, j2 = op
        print op, ptb_words[i1:i2], ms_words[j1:j2] 
        if tag == 'replace':
            # easy case: same number of words --> just substitute
            if (i2 - i1) == (j2 - j1):
                replace_words(ptb_tree, ms_words, i1, i2, j1, j2)
            # harder case: replace ptb_words[i1:i2] by ms_words[j1:j2]
            else:
                delete_words(ptb_tree, i1, i2)
                insert_words(ptb_tree, ms_words, i1, i2, j1, j2)
        elif tag == 'delete':
            delete_words(ptb_tree, i1, i2)
        else:
            # tag = insert; i = i2 in this case
            if i1 == 0 or i1 == len(ptb_words):
                # inserting at end points
                to_insert = ms_words[j1:j2]
                str_mrg = ['(XX ' + w +')' for w in to_insert]
                mrg = '(X ' + ' '.join(str_mrg) + ' )'
                node_to_add = pstree.tree_from_text(mrg)
                if i1 == 0:
                    ptb_tree = merge_trees(node_to_add, ptb_tree)
                else:
                    ptb_tree = merge_trees(ptb_tree, node_to_add)
            else:
                insert_words(ptb_tree, ms_words, i1, i2, j1, j2)
        ptb_tree.check_consistency()

    # initial "errors" in ms_tree compared to "gold" of ptb_tree
    if rev:
        init_errors = parse_errors.get_errors(ptb_tree, ms_tree)
        iters, path = transform_search.greedy_search(ms_tree, ptb_tree, classify)
    else:
        init_errors = parse_errors.get_errors(ms_tree, ptb_tree)
        iters, path = transform_search.greedy_search(ptb_tree, ms_tree, classify)

    # find transformation from ms_tree to ptb_tree
    return init_errors, iters, path, ptb_tree

def gather_output(ptb_mrg, ms_candidates, sent_name_prefix, ms_hypo_dir, \
        draw_tree, rev):
    
    if rev:
        fout = open(sent_name_prefix + '.out_rev', 'w')
    else:
        fout = open(sent_name_prefix + '.out', 'w')

    if draw_tree:
        sent_name = os.path.join(ms_hypo_dir, sent_name_prefix + '_' + 'ptb.ps')
        t = Tree.fromstring(ptb_mrg)
        TreeView(t)._cframe.print_to_file(sent_name)
        if sent_name_prefix in ann_sents:
            ann_mrgs = ann_df[ann_df.sent_id==sent_name_prefix].human_mrg.values
            for i, m in enumerate(ann_mrgs):
                temp = pstree.tree_from_text("(ROOT" + m[1:])
                # treebanks.homogenise_tree(temp)
                treebanks.remove_function_tags(temp)
                t = Tree.fromstring(str(temp))
                sent_name = os.path.join(ms_hypo_dir, sent_name_prefix + '_' + \
                        str(i) + '_' + 'ann.ps')
                TreeView(t)._cframe.print_to_file(sent_name)

    for i, ms_mrg in enumerate(ms_candidates):
        print >> fout, "MS Tree candidate {}".format(i)
        ptb_tree = pstree.tree_from_text(ptb_mrg)
        ms_tree = pstree.tree_from_text(ms_mrg)
        if draw_tree:
            sent_name = os.path.join(ms_hypo_dir, sent_name_prefix + \
                    '_' + str(i) + '_' + 'ms.ps')
            t = Tree.fromstring(ms_mrg)
            TreeView(t)._cframe.print_to_file(sent_name)
        init_errors, iters, path, new_tree = convert_tree(ptb_tree, ms_tree, rev)
        if draw_tree and i == 0:
            # draw the intermediate tree
            sent_name = os.path.join(ms_hypo_dir, sent_name_prefix + \
                    '_' + 'ptb_updated.ps')
            t = Tree.fromstring(str(new_tree))
            TreeView(t)._cframe.print_to_file(sent_name)

        error_count = len(init_errors)
        print >> fout, "{} Initial errors".format(error_count)
        print >> fout, "{} on fringe, {} iterations".format(*iters)
        if path is not None:
            for tree in path[1:]:
                print >> fout, "{} Error:{}".format(str(tree[2]), \
                        tree[1]['classified_type'])
            if len(path) > 1:
                for tree in path:
                    print >> fout, "Step:{}".format(tree[1]['classified_type'])
                    print >> fout, tree[1]
                    print >> fout, render_tree.text_coloured_errors(tree[0], \
                            gold=ptb_tree).strip()

        else:
            print "no path found"
        print >> fout, "\n"
    fout.close()


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Compare two models')
    pa.add_argument('--draw_tree', type=int, default=0, \
            help='set true to draw tree')
    pa.add_argument('--rev', type=int, default=1, \
            help='0: transform from MS candidate to PTB; 1: vice versa')
    pa.add_argument('--turn_file', type=str, default='../2015_B.csv', \
            help='csv file with alignment of 1 speaker side')
    pa.add_argument('--sent_id', type=str, default='B30_6', \
            help='sentence id')
    pa.add_argument('--ms_hypo_dir', type=str, default='../samples',\
            help='directory of ms hypotheses')

    args = pa.parse_args()
    draw_tree = bool(args.draw_tree)
    rev = bool(args.rev)
    turn_file = args.turn_file
    sent_id = args.sent_id
    ms_hypo_dir = args.ms_hypo_dir

    # columns of .csv file:
    cols = ['sent_id', 'ms_sent_raw', 'ms_sent_clean', 'ptb_sent', 'times', \
            'mrg']
    turn_df = pd.read_csv(turn_file, names=cols, sep='\t')
    file_num = os.path.basename(turn_file).split('_')[0]
    sent_name_prefix = file_num + '_' + sent_id
    print sent_name_prefix

    turn_df = turn_df.set_index('sent_id')
    row = turn_df.loc[sent_name_prefix]
    ptb_mrg = row.mrg
    ptb_mrg = "(ROOT"+ ptb_mrg[1:]

    ms_file = os.path.join(ms_hypo_dir, sent_name_prefix + '.ms.bkout')
    ms_candidates = open(ms_file).readlines()
    ms_candidates = [x.strip() for x in ms_candidates]
    ms_candidates = ["(ROOT"+ x[1:] for x in ms_candidates if x]

    gather_output(ptb_mrg, ms_candidates, sent_name_prefix, ms_hypo_dir, \
            draw_tree, rev)


