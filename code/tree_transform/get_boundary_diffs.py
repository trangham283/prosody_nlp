#!/usr/bin/env python

from __future__ import division
import os
import sys
import argparse
import cPickle as pickle
import pandas as pd
import numpy as np
from itertools import groupby
from difflib import SequenceMatcher

def convert_sent_name(x, y):
    y_str = '_' + y + '_'
    ans = x.replace('~', y_str)
    return ans

def name_change(sp_name):
    return sp_name.replace('trans','dat')

def get_speaker(sent_id):
    return sent_id[0]

def get_turn(sent_id):
    return int(sent_id.replace('A','').replace('B',''))

def make_list(tokens):
    if not isinstance(tokens, basestring):
        return tokens
    else:
        all_str = tokens.strip().lstrip("['").rstrip("']")
        all_str = all_str.split()
        all_str = [x.rstrip("',").lstrip("'").rstrip('"').lstrip('"') \
                for x in all_str]
        return all_str

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

def preprocess(project_dir, split):
    # df1: treebank
    f1 = os.path.join(project_dir, split+'_mrg.tsv')
    df1 = pd.read_csv(f1, sep='\t')
    
    # df2: dff
    #if split == 'train':
    #    split_name = ''
    #else:
    #    split_name = '_' + split
    #f2 = os.path.join(project_dir, \
    #        'treebank_disfl_annotation_with_s' + split_name + '.csv')
    f2 = os.path.join(project_dir, \
            'treebank_disfl_annotation_with_s_combine_turns.csv')
    df2 = pd.read_csv(f2, sep=',').fillna(0)
    df2['turn'] = df2.turn.apply(int)

    # clean up df1
    df1['file_num'] = df1['file_id'].apply(lambda x: \
            int(x.rstrip('.dat').lstrip('sw')))
    #df1['tokens'] = df1['sentence'].apply(lambda x: x.strip().split())
    df1['tokens'] = df1['sentence'].apply(get_tokens)
    df1['mrg'] = df1['mrg'].apply(norm_mrg)
    df1 = df1.rename(columns={'file_id': 'file', 'sent_id': 'turn_id', \
            'tree_id': 'sent_num'})
    df1['speaker'] = df1.turn_id.apply(lambda x: x[0])
    df1['turn'] = df1.turn_id.apply(lambda x: int(x[1:]))

    # clean up df2
    df2 = df2.rename(columns={'sentence': 'tokens'})
    df2.tokens = df2.tokens.apply(make_list)
    df2['file_num'] = df2['file'].apply(lambda x: \
            int(x.rstrip('.dat').lstrip('sw')))

    return df1, df2

def preprocess_old(project_dir, split):
    split_name = ''
    if split == 'dev' or split == 'test':
        split_name = '_dev_test'

    # df1: treebank
    f1 = os.path.join(project_dir, split+'.mrg.tsv')
    df1 = pd.read_csv(f1, sep='\t')
    
    # df2: dff
    f2 = os.path.join(project_dir, \
            'treebank_disfl_annotation_with_s' + split_name + '.csv')
    df2 = pd.read_csv(f2, sep=',').fillna(0)
    df2['turn'] = df2.turn.apply(int)

    # clean up df1
    df1['file_num'] = df1['file_id'].apply(lambda x: \
            int(x.rstrip('.dat').lstrip('sw')))
    df1['tokens'] = df1['sentence'].apply(lambda x: x.strip().split())
    df1 = df1.rename(columns={'file_id': 'file'})

    # clean up df2
    df2 = df2.rename(columns={'sentence': 'tokens'})
    df2.tokens = df2.tokens.apply(make_list)
    df2['file_num'] = df2['file'].apply(lambda x: \
            int(x.rstrip('.dat').lstrip('sw')))

    return df1, df2

def get_offset_ids(df):
    offsets = [0]
    for i, row in df.iterrows():
        offsets.append(offsets[i] + len(row.tokens) + 1) #+1 accounting for \n
    return offsets

def print_turn(df1, df2, file_num, speaker):
    prefix = str(file_num) + '_' + speaker 
    finfo = prefix + '.sent_ids.tsv'
    fdff = open(prefix + '.dff', 'w')
    fstr = open(prefix + '.str', 'w')
    fmrg = open(prefix + '.mrg', 'w')
    fidxd = open(prefix + '.dff.idx', 'w')
    fidxm = open(prefix + '.mrg.idx', 'w')
    this_df1 = df1[(df1.speaker == speaker) & (df1.file_num == file_num)]
    this_df2 = df2[(df2.speaker == speaker) & (df2.file_num == file_num)]
    this_df2 = this_df2.sort_values(by=['turn','sent_num']).reset_index()
    dff_toks = this_df2.tokens.values.copy()
    this_df1 = this_df1.sort_values(by=['turn','sent_num']).reset_index()
    #this_df1 = this_df1.sort_values(by='sent_id').reset_index()
    tree_toks = this_df1.tokens.values.copy()

    this_df2['offsets'] = get_offset_ids(this_df2)[:-1]
    this_df1['offsets'] = get_offset_ids(this_df1)[:-1]

    da = dict(zip(this_df2['offsets'], zip(this_df2.turn, this_df2.sent_num, \
            range(len(this_df2)))))
    #db = dict(zip(this_df1['offsets'], zip(this_df1.sent_id, \
    #        range(len(this_df1)))))
    db = dict(zip(this_df1['offsets'], zip(this_df1.turn, this_df1.sent_num, \
            range(len(this_df1)))))
    dav_sorted = [da[k] for k in sorted(da.keys())]
    dbv_sorted = [db[k] for k in sorted(db.keys())]

    seqa = []
    seqb = []
    for s in dff_toks[:]:
        seqa += s
        seqa += ['\n']
    for s in tree_toks[:]:
        seqb += s
        seqb += ['\n']

    list_row = []

    # assuming dff_toks have better segmentation:
    # .get_opcodes returns ops to turn a into b 
    sseq = SequenceMatcher(None, seqa, seqb)
    idx = 0
    for tag, i1, i2, j1, j2 in sseq.get_opcodes():
        if tag != 'equal':
            if tag == 'delete': 
                if seqa[i1:i2] == ['--','--'] or seqa[i1:i2] == ['//']:
                    # not an error
                    continue
            seqa_lo0 = sorted([k for k in da.keys() if k<=i1])[-1]
            seqa_lo1 = sorted([k for k in da.keys() if k<=i2])[-1]
            seqb_lo0 = sorted([k for k in db.keys() if k<=j1])[-1]
            seqb_lo1 = sorted([k for k in db.keys() if k<=j2])[-1]
            
            seqa_toks = seqa[i1:i2]
            seqb_toks = seqb[j1:j2]
            dff_sents = dav_sorted[da[seqa_lo0][-1]:da[seqa_lo1][-1]+1]
            mrg_sents = dbv_sorted[db[seqb_lo0][-1]:db[seqb_lo1][-1]+1]

            list_row.append({'idx' : idx, \
                    'action' : tag, \
                    'dff_toks' : seqa_toks, \
                    'mrg_toks' : seqb_toks, \
                    'dff_sents' : dff_sents, \
                    'mrg_sents' : mrg_sents})

            for turn, sent_num, local_num in mrg_sents:
                sent_str = this_df1[(this_df1.turn == turn) & \
                        (this_df1.sent_num==sent_num)].sentence.values[0]
                item = sent_str + "\n"
                fstr.write(item)
                mrg_str = this_df1[(this_df1.turn == turn) & \
                        (this_df1.sent_num==sent_num)].mrg.values[0]
                item = mrg_str + "\n"
                fmrg.write(item)
                item = str(idx) + "\n"
                fidxm.write(item)

            for turn, sent_num, local_num in dff_sents:
                sent_tokens = this_df2[(this_df2.turn==turn) & \
                        (this_df2.sent_num==sent_num)].tokens.values[0]
                sent_str = ' '.join(sent_tokens)
                item = sent_str + "\n"
                fdff.write(item)
                item = str(idx) + "\n"
                fidxd.write(item)

            idx += 1

            #print ("%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
            #        i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))
            #print "\taffected dff sents: ", \
            #        dav_sorted[da[seqa_lo0][-1]:da[seqa_lo1][-1]+1]
            #print "\taffected tree sents: ", \
            #        dbv_sorted[db[seqb_lo0][-1]:db[seqb_lo1][-1]+1]

    df = pd.DataFrame(list_row)
    df.to_csv(finfo, sep = '\t')
    fstr.close()
    fmrg.close()
    fdff.close()
    fidxd.close()
    fidxm.close()


def process_turn(df1, df2, file_num, speaker):
    split, merge, other = 0, 0, 0
    this_df1 = df1[(df1.speaker == speaker) & (df1.file_num == file_num)]
    this_df2 = df2[(df2.speaker == speaker) & (df2.file_num == file_num)]

    this_df2 = this_df2.sort_values(by=['turn','sent_num']).reset_index()
    dff_toks = this_df2.tokens.values.copy()
    #this_df1 = this_df1.sort_values(by='sent_id').reset_index()
    this_df1 = this_df1.sort_values(by=['turn','sent_num']).reset_index()
    tree_toks = this_df1.tokens.values.copy()

    this_df2['offsets'] = get_offset_ids(this_df2)[:-1]
    this_df1['offsets'] = get_offset_ids(this_df1)[:-1]

    da = dict(zip(this_df2['offsets'], zip(this_df2.turn, this_df2.sent_num)))
    db = dict(zip(this_df1['offsets'], zip(this_df1.turn, this_df1.sent_num)))
    #db = dict(zip(this_df1['offsets'], this_df1.sent_id))

    seqa = []
    seqb = []
    for s in dff_toks[:]:
        seqa += s
        seqa += ['\n']
    for s in tree_toks[:]:
        seqb += s
        seqb += ['\n']

    # assuming dff_toks have better segmentation:
    # .get_opcodes returns ops to turn a into b 
    sseq = SequenceMatcher(None, seqa, seqb)
    for tag, i1, i2, j1, j2 in sseq.get_opcodes():
        if tag != 'equal':
            if tag == 'delete': 
                if i2-i1 == 1 and seqa[i1:i2][0] == '\n':
                    merge += 1
                    print "MERGE"
                elif seqa[i1:i2] == ['--','--'] or seqa[i1:i2] == ['//']:
                    # not an error
                    continue
            elif tag == 'insert' and j2-j1 == 1 and seqb[j1:j2][0] == '\n':
                split += 1
                print "SPLIT"
            elif tag == 'replace':
                if seqa[i1:i2] == ['--','--'] and seqb[j1:j2] == ['\n']:
                    split += 1
                    print "SPLIT"
                if seqa[i1:i2][0] == ''.join(seqb[j1:j2]):
                    # contractions
                    continue
                else:
                    other += 1
                    print "OTHER"
            else:
                other += 1
                print "OTHER"
                #seqa_lo0 = sorted([k for k in da.keys() if k<=i1])[-1]
                #seqa_lo1 = sorted([k for k in da.keys() if k<=i2])[-1]
                #seqb_lo0 = sorted([k for k in db.keys() if k<=j1])[-1]
                #seqb_lo1 = sorted([k for k in db.keys() if k<=j2])[-1]
                #print file_num, speaker
                #print ("\t%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
                #        i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))
                #print "\taffected dff sents: ", da[seqa_lo0], da[seqa_lo1]
                #print "\taffected tree sents: ", db[seqb_lo0], db[seqb_lo1]

            seqa_lo0 = sorted([k for k in da.keys() if k<=i1])[-1]
            seqa_lo1 = sorted([k for k in da.keys() if k<=i2])[-1]
            seqb_lo0 = sorted([k for k in db.keys() if k<=j1])[-1]
            seqb_lo1 = sorted([k for k in db.keys() if k<=j2])[-1]
            print ("%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
                    i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))
            print "\taffected dff sents: ", da[seqa_lo0], da[seqa_lo1]
            print "\taffected tree sents: ", db[seqb_lo0], db[seqb_lo1]

            #if seqa_lo0 != seqa_lo1: 
            #    print "Boundary Deletion OR replaced word spread over sents"
            #    print ("%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
            #            i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))
            #if seqb_lo0 != seqb_lo1: 
            #    print "Boundary Insertion OR replaced word spread over sents"
            #    print ("%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
            #            i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))

    return len(this_df2), len(this_df1), split, merge, other


def process_all(df1, df2, statname):
    files1 = set(df1.file_num)
    files2 = set(df2.file_num)
    files = files1.intersection(files2)
    #print len(files), len(files1), len(files2)
    #print len(files1.difference(files)), len(files2.difference(files))
    all_stats = []
    for f_num in files:
        for speaker in ['A', 'B']:
            outs = process_turn(df1, df2, f_num, speaker)
            #dff_len, mrg_len, split, merge, other = outs
            all_stats.append(outs)
    all_stats = np.array(all_stats)
    print "total number of: "
    print "\tdff SUs, tree sents, splits, merge, other"
    print "\t", np.sum(all_stats, 0)
    #print np.min(all_stats, 0)
    #print np.max(all_stats, 0)
    #print np.mean(all_stats, 0)
    #print np.median(all_stats, 0)
    #pickle.dump(all_stats, open(statname + '_boundary_stats_updated.pickle', \
    #        'w'))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Compare two models')
    pa.add_argument('--project_dir', \
            default='/g/ssli/projects/disfluencies/switchboard_data', \
            help='project directory')
    pa.add_argument('--file_num', type=int, default=2005, \
            help='file number, if aligning only 1 file')
    pa.add_argument('--speaker', default='A', \
            help='speaker, if aligning only 1 file')
    pa.add_argument('--split', default='dev_test',\
            help='split: dev_test or train')

    args = pa.parse_args()

    df1, df2 = preprocess(args.project_dir, args.split)
    #process_all(df1, df2, args.split)
    
    # inspect files
    outs = process_turn(df1, df2, args.file_num, args.speaker)
    print outs
    #print_turn(df1, df2, args.file_num, args.speaker)


