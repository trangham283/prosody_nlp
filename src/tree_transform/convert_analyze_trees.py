#!/usr/bin/env python
from parse_analyzer import render_tree, init, treebanks, parse_errors, nlp_eval
from parse_analyzer import pstree, transform_search, head_finder, tree_transform
from tree_convert_utils import *
import sys, argparse, os
from parse_analyzer.classify_english import classify
from collections import defaultdict
from StringIO import StringIO
from difflib import SequenceMatcher
from nltk.tree import Tree
from nltk.draw.tree import TreeView
import pandas as pd 
import cPickle as pickle
import numpy as np

# dictionary of most common tags
# tag_counts = pickle.load(open('tag_counts.pickle'))

# file with limited human annotation
# this has at least the following columns:
# [sent_id, human_mrg]
ann_df = pd.read_csv('human-ann.tsv', sep='\t')
ann_sents = set(ann_df.sent_id)

def make_sent_id(file_num, speaker, turn, sent_num):
    sent_id = "{}_{}{}_{}".format(file_num, speaker, turn, sent_num)
    return sent_id

def is_in_set(left, running_set):
    i = 0
    for item in running_set:
        if left == item[0]:
            return i
        i += 1
    return -1

def gather_output_singles(ptb_mrg, ms_candidates, ms_scores, \
        sent_name_prefix, ms_hypo_dir, out_dir, draw_tree, rev, fcomp):
    
    if rev:
        fout = open(os.path.join(out_dir, sent_name_prefix + '.out_rev'), 'w')
    else:
        fout = open(os.path.join(out_dir, sent_name_prefix + '.out'), 'w')
   
    ptb_tree = pstree.tree_from_text(ptb_mrg)
    if draw_tree:
        sent_name = os.path.join(out_dir, sent_name_prefix + '_' + 'ptb.ps')
        t = Tree.fromstring(ptb_mrg)
        TreeView(t)._cframe.print_to_file(sent_name)
   
    if sent_name_prefix in ann_sents:
        ann_mrgs = ann_df[ann_df.sent_id==sent_name_prefix].human_mrg.values
        if len(ann_mrgs) > 1: 
            print "More than 1 ann sent: ", sent_name_prefix 
            
        for i, m in enumerate(ann_mrgs):
            ann_tree = pstree.tree_from_text("(ROOT " + m[1:])
            treebanks.homogenise_tree(ann_tree)
            treebanks.remove_function_tags(ann_tree)
            if draw_tree:
                t = Tree.fromstring(str(ann_tree))
                sent_name = os.path.join(out_dir, sent_name_prefix + '_' + \
                        str(i) + '_' + 'ann.ps')
                TreeView(t)._cframe.print_to_file(sent_name)
        #m13, g13, t13, _, _ = parse_errors.counts_for_prf(ptb_tree, ann_tree)
        #p13, r13, f13 = nlp_eval.calc_prf(m13, g13, t13)
        #print >> fout, "Comparing PTB and Human: f1 = {}".format(f13)

    for i, ms_mrg in enumerate(ms_candidates):
        print >> fout, "MS Tree candidate {}".format(i)
        print >> fout, "Parse tree score: {}".format(ms_scores[i])
        # Need to reconstruct ptb_tree because changes were made in place
        ptb_tree = pstree.tree_from_text(ptb_mrg)
        ms_tree = pstree.tree_from_text(ms_mrg)
        sent_len = len(ms_tree.word_yield(as_list=True))
        
        # counts_for_prf(test, gold)
        if sent_name_prefix in ann_sents:
            m23, g23, t23, _, _ = parse_errors.counts_for_prf(ms_tree, ann_tree)
            p23, r23, f23 = nlp_eval.calc_prf(m23, g23, t23)
            print >> fout, "Comparing MS Tree and Human: f1 = {}".format(f23)
        else:
            f23 = np.nan
        if draw_tree:
            sent_name = os.path.join(out_dir, sent_name_prefix + \
                    '_' + str(i) + '_' + 'ms.ps')
            t = Tree.fromstring(ms_mrg)
            TreeView(t)._cframe.print_to_file(sent_name)

        init_errors, iters, path, new_tree, f12, correct_tags = convert_tree(\
                ptb_tree, ms_tree, rev)

        if draw_tree and i == 0:
            # draw the intermediate tree
            sent_name = os.path.join(out_dir, sent_name_prefix + \
                    '_' + 'ptb_updated.ps')
            t = Tree.fromstring(str(new_tree))
            TreeView(t)._cframe.print_to_file(sent_name)
        
        print >> fout, "Comparing PTB and MS Tree: f1 = {}".format(f12)
        print >> fout, "{} correct tags".format(correct_tags)
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
        item = "{}_{}\t{}\t{}\t{}\t{}\t{}\t{}".format(sent_name_prefix, i, \
            ms_scores[i], f23, error_count, iters[-1], correct_tags, sent_len) 
        print >> fcomp, item
    fout.close()
  
def process_singles(fcompname, sent_file, ms_hypo_dir, out_dir, \
        draw_tree, rev, ps_suffix):
    fcomp = open(fcompname, 'w')
    print >> fcomp, "sent_id\ttree_score\tf1_ms_vs_human\tinit_errors\titers\tPOS_match\tsent_len"
    
    # columns of .csv file (previously in something like samples/3727_B.csv):
    #cols = ['sent_id', 'ms_sent_raw', 'ms_sent_clean', 'ptb_sent', 'times', \
    #        'mrg']
    # columns of .tsv file (singles for now):
    cols = ['sent_id', 'ms_sent', 'ptb_sent', 'comb_ann', 'times', \
            'mrg']
    file_examples = open(sent_file).readlines()
    file_examples = [x.strip() for x in file_examples]
    file_list = {}
    for s in file_examples:
        file_num, turn_id, sent_id = s.split('_')
        speaker = turn_id[0]
        if (int(file_num), speaker) not in file_list:
            file_list[(int(file_num), speaker)] = []
        file_list[(int(file_num), speaker)].append( \
                "{}_{}".format(turn_id, sent_id))
    file_set = set(file_list.keys())
    print file_list

    for file_num, speaker in file_set:
        #turn_file = os.path.join(ms_hypo_dir, \
        #        "{}_{}.csv".format(file_num, speaker))
        turn_file = os.path.join(ms_hypo_dir, \
                "{}_{}_{}.tsv".format(file_num, speaker, ps_suffix))
        turn_df = pd.read_csv(turn_file, names=cols, sep='\t')
        turn_df = turn_df.set_index('sent_id')
        for sent_id in file_list[(int(file_num), speaker)]:
            sent_name_prefix = "{}_{}".format(file_num, sent_id)
            print sent_name_prefix
            row = turn_df.loc[sent_name_prefix]
            ptb_mrg = row.mrg
            ptb_mrg = "(ROOT"+ ptb_mrg[1:]

            ms_score_file = os.path.join(ms_hypo_dir, sent_name_prefix + \
                    '.ms.out_with_score')
            ms_scores = open(ms_score_file).readlines()
            ms_scores = [x.strip() for x in ms_scores]
            score_and_sent = [x.split('\t') for x in ms_scores if x]
            scores = [float(x[0]) for x in score_and_sent]
            ms_candidates = [x[1] for x in score_and_sent]
            ms_candidates = ["(ROOT "+ x[1:] for x in ms_candidates]

            gather_output_singles(ptb_mrg, ms_candidates, scores, \
                    sent_name_prefix, ms_hypo_dir, out_dir, \
                    draw_tree, rev, fcomp)
    fcomp.close()


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Perform tree transformations and analyze')
    pa.add_argument('--draw_tree', type=int, default=0, \
            help='set true to draw tree')
    pa.add_argument('--rev', type=int, default=1, \
            help='0: transform from MS candidate to PTB; 1: vice versa')
    pa.add_argument('--ms_hypo_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_info',\
            help='directory of ms hypotheses')
    pa.add_argument('--out_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_tree_out',\
            help='directory to dump output to')
    pa.add_argument('--ps_suffix', type=str, default='updated_pairs', \
            help='suffix of paired files: singles OR updated_pairs')
    pa.add_argument('--sent_file', type=str, \
            default='debug-sents-small-ps.txt', \
            help='list of sentences to analyze')
    pa.add_argument('--fcompname', type=str, default='debug-ps.tsv',\
            help='file to collect results')

    # previously both ms_hypo_dir and out_dir were "samples"
    args = pa.parse_args()
    draw_tree = bool(args.draw_tree)
    rev = bool(args.rev)
    sent_file = args.sent_file
    ms_hypo_dir = args.ms_hypo_dir
    out_dir = args.out_dir
    fcompname = args.fcompname
    ps_suffix = args.ps_suffix

    process_singles(fcompname, sent_file, ms_hypo_dir, out_dir, \
                draw_tree, rev, ps_suffix)

