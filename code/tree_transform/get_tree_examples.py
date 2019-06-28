#!/usr/bin/env python

from __future__ import division
import os
import sys
import argparse
import cPickle as pickle
import pandas as pd
import numpy as np
from glob import glob
from itertools import groupby
from difflib import SequenceMatcher
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from termcolor import colored

columns = ['ms_id', 'ptb_id', 'start_time', 'end_time', 'alignment', \
        'ptb_word', 'ms_word']
ERR = ['<SUB>', '<DEL>', '<INS>']
OTHER = ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]

def read_ms_file(ms_data_dir, file_num, speaker):
    turn_stats = []
    split_num = str(int(file_num/1000))
    file_name = os.path.join(ms_data_dir, split_num, 'sw' + str(file_num) + \
            speaker + '-ms98-a-penn.text')
    df = pd.read_csv(file_name, sep = '\t', names = columns)
    df['has_error'] = df.alignment.apply(lambda x: x in ERR)
    return df
    
def get_tokens(sentence):
    if not isinstance(sentence, str):
        toks = ['EMPTY_TREE']
    else:
        toks = sentence.strip().split()
    return toks

def norm_mrg(mrg):
    if not isinstance(mrg, str):
        return 'EMPTY_TREE'
    else:
        return mrg

def norm_laughter(word):
    if '[laughter-' in word:
        word = word.lstrip('[laughther').rstrip(']').lstrip('-')
    return word

def unroll_mrg_toks(df_mrg):
    list_row = []
    for i, row in df_mrg.iterrows():
        tokens = row.ptb_word
        for t in tokens:
            list_row.append({'file_num': row.file_num, \
                    'speaker': row.speaker, \
                    'ptb_word': t.lower(), \
                    'sent_id': row.sent_id, \
                    'tree_id': row.tree_id})
    df = pd.DataFrame(list_row)
    return df

def match_idx(info):
    tag, i1, i2, j1, j2 = info
    ms_range = range(i1, i2)
    ptb_range = range(j1, j2)
    if tag == 'equal':
        match = zip(ptb_range, ms_range)
    # don't need this 
    elif tag == 'delete':
        # ptb_range is empty (j1=j2): match to fractions between [j1-1, j]
        lo = j1 - 1
        hi = j1
        num_points = len(ms_range) + 2
        tile = np.linspace(lo, hi, num_points)[1:-1]
        match = zip(tile, ms_range)
    elif tag == 'insert':
        # ms_range is empty (i1=i2): match ptb indices to i2
        tile = [i2]*len(ptb_range)
        match = zip(ptb_range, tile)
    else:
        # replace
        match = [(j1, i1), (j2, i2)]
        for i, k in enumerate(ptb_range[1:]):
            if (i1+i+1) < i2: match.append((k, i1+i+1))
            else: match.append((k, i2))
    return match

def preprocess_mrg(project_dir, split):
    # df from Penn treebank
    fmrg = os.path.join(project_dir, split + '_mrg.tsv')
    df_mrg_all = pd.read_csv(fmrg, sep='\t')
    df_mrg_all['speaker'] = df_mrg_all.sent_id.apply(lambda x: x[0])
    df_mrg_all['file_num'] = df_mrg_all.file_id.apply(lambda x: int(x[2:]))
    #df_mrg = df_mrg_all[(df_mrg_all.file_num == file_num)]
    #df_mrg = df_mrg[(df_mrg.speaker == speaker)]
    #df_mrg['mrg'] = df_mrg['mrg'].apply(norm_mrg)
    #df_mrg['ptb_word'] = df_mrg['sentence'].apply(get_tokens)
    #df_mrg_unrolled = unroll_mrg_toks(df_mrg)
    return df_mrg_all

def preprocess_ms(ms_data_dir, file_num, speaker):
    # df from MS-State
    #df_msA = read_ms_file(ms_data_dir, file_num, 'A')
    #df_msB = read_ms_file(ms_data_dir, file_num, 'B')
    #df_ms = pd.concat([df_msA, df_msB])
    df_ms = read_ms_file(ms_data_dir, file_num, speaker)
    df_ms['sent_id'] = df_ms.ptb_id.apply(lambda x: x.replace('.', ''))
    df_ms['ms_word'] = df_ms.ms_word.apply(norm_laughter)
    df_ms['ptb_word'] = df_ms.ptb_word.apply(lambda x: x.lower())
    for other_word in OTHER:
        df_ms = df_ms[(df_ms.ms_word != other_word)]
    return df_ms


def align_msptb(df_ms, df_mrg_all, file_num, speaker):
    df_mrg = df_mrg_all[(df_mrg_all.file_num == file_num)]
    df_mrg = df_mrg[(df_mrg.speaker == speaker)]
    df_mrg['mrg'] = df_mrg['mrg'].apply(norm_mrg)
    df_mrg['ptb_word'] = df_mrg['sentence'].apply(get_tokens)
    #print df_mrg.head(2)
    df_mrg_unrolled = unroll_mrg_toks(df_mrg)

    ms_toks = df_ms.ptb_word.values.copy()
    df_ms['ms_tok_id'] = range(len(ms_toks))
    ptb_toks = df_mrg_unrolled.ptb_word.values.copy()
    df_mrg_unrolled['ptb_tok_id'] = range(len(ptb_toks))
    
    # .get_opcodes returns ops to turn a into b 
    sseq = SequenceMatcher(None, ms_toks, ptb_toks)
    ptb2ms = []
    for info in sseq.get_opcodes():
        match = match_idx(info)
        ptb2ms.append(match)
    ptb2ms = [item for sublist in ptb2ms for item in sublist]
    ptb2ms = dict(ptb2ms)
    return ptb2ms, df_mrg, df_mrg_unrolled 
                
def get_error_sents(df):
    df_err = df[df.alignment.isin(ERR)]
    err_sents = set([])
    for sent_id, df_by_turn in df_err.groupby('sent_id'):
        if len(df_by_turn) > 5:
            err_sents.add(sent_id)
    return err_sents

# Dealing with splitting contractions, possessives, and things like "gonna"
# special cases: originally split_case2.txt
tok_map = {
        "[jere's/there's]": ["there", "'s"], 
        "y'all": ["you", "all"],
        "gonna": ["going", "to"],
        "wanna": ["want", "to"],
        "ain't": ["ai", "n't"],
        "aren't": ["are", "n't"], 
        "can't": ["ca", "n't"],
        "couldn't": ["could", "n't"],
        "-[d]idn't": ["-[d]id", "n't"],
        "didn't": ["did", "n't"],
        "doesn't": ["does", "n't"],
        "-[d]on't": ["-[d]o", "n't"], 
        "don't": ["do", "n't"],
        "hadn't": ["had", "n't"],
        "hasn't": ["has", "n't"],
        "haven't": ["have", "n't"],
        "isn't": ["is", "n't"],
        "shouldn't": ["should", "n't"],
        "wasn't": ["was", "n't"],
        "weren't": ["were", "n't"],
        "won't": ["wo", "n't"],
        "wouldn't": ["would", "n't"],
        "[doen't/doesn't]": ["does", "n't"],
        "[in't/isn't]": ["is", "n't"],
        "[it'n/isn't]": ["is", "n't"],
        "[what'n/wasn't]": ["was", "n't"],
        "[wan't/wasn't]": ["was", "n't"],
        "[non't/don't]": ["do", "n't"],
        "cannot": ["can", "not"],
        "---" : []
        }

# split1 words: ending lasts 1 phone
split1_file = '/homes/ttmt001/transitory/prosodic-anomalies/split_case1.txt'
split1_lines = open(split1_file).readlines()
split1_words = [s.rstrip() for s in split1_lines]
split1_words = [(s, s.split("'")) for s in split1_words]

for k, v in split1_words:
    tok_map[k] = [v[0], "'" + v[1]]

def clean_up(toks):
    new_toks = []
    for t in toks:
        t = t.lower()
        if t in tok_map:
            new_toks += tok_map[t]
        else:
            new_toks.append(t)
    return new_toks

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Get sentence-level examples to parse')
    pa.add_argument('--project_dir', \
            default='/g/ssli/projects/disfluencies/switchboard_data', \
            help='project directory')
    pa.add_argument('--ms_data_dir', \
            default='/s0/ttmt001/speech_parsing/ms_alignment/data/alignments', \
            help='ms data directory')
    pa.add_argument('--draw_tree', type=bool, default=False, \
            help='set true to draw tree')
    pa.add_argument('--sent_file', type=str, default=None, \
            help='list of sentence files')
    pa.add_argument('--out_dir', type=str, default='samples', \
            help='output directory')
    pa.add_argument('--split', default='train',\
            help='split: dev_test or train')

    args = pa.parse_args()

    sent_file = args.sent_file
    out_dir = args.out_dir

    df_mrg_all = preprocess_mrg(args.project_dir, args.split)
    file_examples = open(sent_file).readlines()
    file_examples = [x.strip() for x in file_examples]
    file_list = {}
    for s in file_examples:
        file_num, turn_id, sent_id = s.split('_')
        speaker = turn_id[0]
        if (int(file_num), speaker) not in file_list:
            file_list[(int(file_num), speaker)] = set([])
        file_list[(int(file_num), speaker)].add(turn_id)
    file_set = set(file_list.keys())
    print file_list

    for file_num, speaker in file_set:
        print file_num, speaker
        df_ms = preprocess_ms(args.ms_data_dir, file_num, speaker)
        ptb2ms, df_mrg, df_mrg_unrolled = align_msptb(df_ms, df_mrg_all, \
                file_num, speaker)
        
        out_name = os.path.join(out_dir, str(file_num) +'_'+ speaker + '.csv')
        fout = open(out_name, 'w')
        
        for sent_id in file_list[(file_num, speaker)]:
            sent_name_prefix = str(file_num) + '_' + sent_id.replace('@','')
            print sent_name_prefix
            mrg_sents = df_mrg[df_mrg.sent_id == \
                    sent_id.replace('@','')].mrg.values
            str_sents = df_mrg[df_mrg.sent_id == \
                    sent_id.replace('@','')].sentence.values
            inter_df = df_mrg_unrolled[df_mrg_unrolled.sent_id == sent_id]
            df_ms_by_tok = df_ms.set_index('ms_tok_id')
            for tree_id, df_for_tree in inter_df.groupby('tree_id'):
                f_ms = open(os.path.join(out_dir, sent_name_prefix + \
                        '_' + str(tree_id) + '.ms'), 'w')
                start_ptb_idx = df_for_tree.ptb_tok_id.min()
                end_ptb_idx = df_for_tree.ptb_tok_id.max()
                start_ms_idx = ptb2ms[start_ptb_idx]
                end_ms_idx = ptb2ms[end_ptb_idx]
                ms_sent_df = df_ms_by_tok.iloc[start_ms_idx:end_ms_idx+1]
                ms_toks = ms_sent_df.ms_word.values
                ms_sent_raw = " ".join(ms_toks)
                ms_toks = clean_up(ms_toks)
                ms_sent_clean = " ".join(ms_toks)
                f_ms.write(ms_sent_clean)
                print tree_id
                #print "\t", ms_sent_df[['sent_id', 'start_time', 'end_time', \
                #        'alignment', 'ms_word', 'ptb_word']]
                #print
                item =  sent_name_prefix + '_' + str(tree_id) + "\t" + \
                        ms_sent_raw + "\t" + ms_sent_clean + "\t" + \
                        str_sents[tree_id] + "\t" + \
                        str(ms_sent_df.start_time.min()) + "; " + \
                        str(ms_sent_df.end_time.max()) + "\t" + \
                        mrg_sents[tree_id]
                f_ms.close()
                fout.write(item + "\n")
                if args.draw_tree:
                    sent_name = os.path.join(out_dir, sent_name_prefix + '_' + \
                        str(tree_id) + '.ps')
                    t = Tree.fromstring(mrg_sents[tree_id])
                    TreeView(t)._cframe.print_to_file(sent_name)
        fout.close()

# columns of .csv file:
# ['sent_id', 'ms_sent_raw', 'ms_sent_clean', 'ptb_sent', 'times', 'mrg']


