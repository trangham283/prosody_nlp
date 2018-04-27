#!/usr/bin/env python
from parse_analyzer import render_tree, init, treebanks, parse_errors, nlp_eval
from parse_analyzer import pstree, transform_search, head_finder, tree_transform
from tree_convert_utils import *
import sys, argparse, os, glob
from parse_analyzer.classify_english import classify
from collections import defaultdict
from StringIO import StringIO
from difflib import SequenceMatcher
from nltk.tree import Tree
from nltk.draw.tree import TreeView
import pandas as pd 
import cPickle as pickle

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

# Produce intermediate trees from PTB versions, updated according to MS words
def align_pairs(file_num, speaker, ms_hypo_dir):
    turn_file = os.path.join(ms_hypo_dir, \
            "{}_{}_pairs.pickle".format(file_num, speaker))
    data = pickle.load(open(turn_file))
    ms_set = data['ms_sents'].keys()
    ptb_set = data['ptb_sents'].keys()
    pairs = data['pairs']
    temp_mrg = {}

    sides = []
    for a in ms_set:
        turn = int(a.split('_')[1][1:])
        sent_num = int(a.split('_')[2])
        ms_side = [x[0] for x in pairs if (turn, sent_num) in x[0]]
        ms_side = [item for sublist in ms_side for item in sublist]
        ms_side = [x for x in ms_side if \
                make_sent_id(file_num, speaker,*x) in data['ms_sents']]
        ms_side = set([x for x in ms_side if \
                data['ms_sents'][make_sent_id(file_num, speaker,*x)][0]])
        ptb_side = [x[1] for x in pairs if (turn, sent_num) in x[0]]
        ptb_side = [item for sublist in ptb_side for item in sublist]
        ptb_side = set([x for x in ptb_side if \
                data['ptb_sents'][make_sent_id(file_num, speaker,*x)][0]])
        idx = is_in_set(ms_side, sides)
        if idx > -1:
            ptb_side = ptb_side.union(sides[idx][1])
            sides[idx] = [ms_side, ptb_side]
        else:
            sides.append([ms_side, ptb_side])
    
    for ms_side, ptb_side in sides:
        #print ms_side, ptb_side
        if not ms_side or not ptb_side: continue
        ms_tokens = []
        offsets = []
        offset = 0
        for turn, sent_num in sorted(ms_side):
            this_id = make_sent_id(file_num, speaker, turn, sent_num)
            this_str = data['ms_sents'][this_id][0]
            this_tokens = this_str.split()
            ms_tokens += this_tokens
            offsets.append(offset)
            offset += len(this_tokens)

        ptb_trees = [] 
        i = 0
        for turn, sent_num in sorted(ptb_side):
            this_id = make_sent_id(file_num, speaker, turn, sent_num)
            # sometimes PTB sentences starts with "(S" instead of "( "
            if data['ptb_sents'][this_id][1][:2] == "(S":
                this_mrg = "(ROOT (S "+data['ptb_sents'][this_id][1][2:]+")"
            else:
                this_mrg = "(ROOT " + data['ptb_sents'][this_id][1][1:]
            this_tree = pstree.tree_from_text(this_mrg)
            if i > 0:
                this_tree = merge_trees(ptb_trees[i-1], this_tree)
            ptb_trees.append(this_tree)
            i += 1

        ptb_words = this_tree.word_yield(as_list=True)
        ptb_tokens = [x.lower() for x in ptb_words]
        if ptb_tokens == ms_tokens:
            updated_ptb_tree = this_tree.clone()
        else: 
            sseq = SequenceMatcher(None, ptb_tokens, ms_tokens)    
            ops = []
            for info in sseq.get_opcodes():
                if info[0] != 'equal': ops.append(info)
            for op in ops[::-1]:
                tag, i1, i2, j1, j2 = op
                if tag == 'replace':
                    # easy case: same number of words --> just substitute
                    if (i2 - i1) == (j2 - j1):
                        replace_words(this_tree, ms_tokens, i1, i2, j1, j2)
                    # harder case: replace ptb_words[i1:i2] by ms_token[j1:j2]
                    else:
                        if (i2-i1) == len(ptb_tokens):
                            # replacing the whole sentence
                            to_insert = ms_tokens[j1:j2]
                            str_mrg = ['(' + look_up_tag(w) + ' ' + w +')' \
                                    for w in to_insert]
                            mrg = '(ROOT (S ' + ' '.join(str_mrg) + ' ) )'
                            this_tree = pstree.tree_from_text(mrg)
                        else:
                            delete_words(this_tree, i1, i2)
                            this_tree = insert_words(this_tree, ms_tokens, \
                                    i1, i2, j1, j2)
                elif tag == 'delete':
                    delete_words(this_tree, i1, i2)
                else:
                    this_tree = insert_words(this_tree, ms_tokens, i1, i2, \
                            j1, j2)
                this_tree.check_consistency()
        
        # now split the MS-sentence of interest if needed
        off_idx = len(ms_side) - 1
        for turn, sent_num in sorted(ms_side)[::-1]:
            ms_sent_id = make_sent_id(file_num, speaker, turn, sent_num)
            if off_idx > 0:
                this_tree, updated_ptb_tree = split_tree(this_tree, \
                    offsets[off_idx])
                temp_mrg[ms_sent_id] = updated_ptb_tree
            else:
                temp_mrg[ms_sent_id] = this_tree
            off_idx -= 1

    # note that temp trees here already have "(ROOT ..."
    return temp_mrg, data

def write_output(temp_mrg, data, out_name):
    # sent_id ms_sent ptb_sent    comb_ann    times   mrg
    list_row = []
    for k in temp_mrg.keys():
        ms_sent = data['ms_sents'][k][0]
        ptb_sent = ' '.join(temp_mrg[k].word_yield(as_list=True)) 
        comb_ann = data['ms_sents'][k][1]
        start_time = data['ms_sents'][k][2]
        end_time = data['ms_sents'][k][3]
        times = "{0};{1}".format(start_time, end_time)
        mrg = str(temp_mrg[k]).replace('ROOT','')
        list_row.append({'sent_id': k, \
                'ms_sent': ms_sent, \
                'ptb_sent': ptb_sent, \
                'comb_ann': comb_ann, \
                'times': times, \
                'mrg': mrg})
    df = pd.DataFrame(list_row)
    df.to_csv(out_name, sep='\t', index=False, header=True, \
        columns=["sent_id", "ms_sent", "ptb_sent", "comb_ann", "times", "mrg"])

# TEMP PATCH
todo = open('check_files.txt').readlines()
todo = [x.strip() for x in todo]
todo = set([int(x.split()[0]) for x in todo])

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Merge/Split cases with mismatched sentence boundaries')
    pa.add_argument('--ms_hypo_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_info',\
            help='directory of ms hypotheses')
    pa.add_argument('--file_num', default=2005, type=int, \
            help='file number, 0 for all')
    pa.add_argument('--speaker', default='A', type=str, \
            help='speaker')
    pa.add_argument('--out_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_info', \
            help='output directory')

    # previously both ms_hypo_dir and out_dir were "samples"
    args = pa.parse_args()

    file_num = args.file_num
    speaker = args.speaker
    out_dir = args.out_dir
    ms_hypo_dir = args.ms_hypo_dir

    if file_num != 0:
        temp_mrg, data = align_pairs(file_num, speaker, ms_hypo_dir)
        out_name = os.path.join(out_dir, \
            "{0}_{1}_updated_pairs.tsv".format(file_num, speaker))
        write_output(temp_mrg, data, out_name)
    else:
        all_files = glob.glob(ms_hypo_dir + "/*_pairs.pickle")
        for f in all_files:
            fname = os.path.basename(f).split('_')
            file_num = int(fname[0])
            # skip done files
            if file_num not in todo: continue
            speaker = fname[1]
            print file_num, speaker
            try:
                temp_mrg, data = align_pairs(file_num, speaker, ms_hypo_dir)
                out_name = os.path.join(out_dir, \
                        "{0}_{1}_updated_pairs.tsv".format(file_num, speaker))
                write_output(temp_mrg, data, out_name)
            except:
                print "Couldn't process file", file_num, speaker
                continue




