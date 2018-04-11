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
from termcolor import colored

# MS-State alignment columns
ERR = ['<SUB>', '<DEL>', '<INS>']
OTHER = ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]

def make_list(tokens):
    if not isinstance(tokens, basestring):
        return tokens
    else:
        all_str = tokens.strip().lstrip("['").rstrip("']")
        all_str = all_str.split()
        all_str = [x.rstrip("',").lstrip("'").rstrip('"').lstrip('"') \
                for x in all_str]
        all_str = [x for x in all_str if x != "None"]
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

def norm_laughter(word):
    if '[laughter-' in word:
        word = word.lstrip('[laughter').rstrip(']').lstrip('-')
    return word

# load treebank mrg dataframe
# preprocess filenum and speaker columns
def preprocess_mrg(project_dir, split, ptb_suffix):
    # df from Penn treebank
    fmrg = os.path.join(project_dir, split + ptb_suffix)
    df_mrg_all = pd.read_csv(fmrg, sep='\t')
    df_mrg_all['speaker'] = df_mrg_all.sent_id.apply(lambda x: x[0])
    df_mrg_all['file_num'] = df_mrg_all.file_id.apply(lambda x: int(x[2:]))
    df_mrg_all['turn'] = df_mrg_all.sent_id.apply(lambda x: int(x[1:]))
    df_mrg_all = df_mrg_all.rename(columns={'tree_id': 'sent_num'})
    return df_mrg_all

# file with "gold" sentence boundaries (slash units)
def preprocess_comb(project_dir, comb_suffix):
    # columns in comb file
    #cols = ['speaker', 'turn', 'sent_num', 'file', 'sentence', \
    #        'ms_sentence', 'comb_sentence', 'names', 'ms_names', \
    #        'comb_ann', 'tags']
    fcomb = os.path.join(project_dir, comb_suffix)
    df_comb = pd.read_csv(fcomb, sep='\t').fillna(0)
    df_comb['turn'] = df_comb.turn.apply(int)
    df_comb = df_comb.rename(columns={'sentence': 'ptb_tokens', \
            'ms_sentence': 'ms_tokens', \
            'names': 'ptb_tok_id', 'ms_names': 'ms_tok_id'})
    df_comb['file_num'] = df_comb['file'].apply(lambda x: \
            int(x.rstrip('.trans').lstrip('sw')))
    return df_comb


switched = set([2010, 2027, 2072, 2073, 2130, 2171, 2177, 2247, 2279, 2290, 
        2305, 2366, 2372, 2405, 2434, 2485, 2501, 2521, 2527, 2533, 2539, 
        2566, 2593, 2617, 2627, 2658, 2789, 2792, 2858, 2913, 2932, 2970, 
        3012, 3040, 3088, 3096, 3130, 3131, 3138, 3140, 3142, 3144, 3146, 
        3148, 3154,
        2006, 2064, 2110, 2235, 2262, 2292, 2303, 2339, 2476, 2514, 2543, 
        2576, 2616, 2631, 2684, 2707, 2794, 2844, 2854, 2930, 2954, 2955, 
        2960, 2963, 2968, 2981, 2983, 2994, 2999, 3000, 3013, 3018, 3039, 
        3050, 3061, 3077, 3136, 3143, 3405])
def preprocess_ms(project_dir, ms_dir_midfix, file_num, speaker):
    columns = ['ms_id', 'ptb_id', 'start_time', 'end_time', 'alignment', \
        'ptb_word', 'ms_word', 'uw_tok_id']
    split_num = str(int(file_num/1000))
    ms_data_dir = os.path.join(project_dir, ms_dir_midfix) 
    if file_num in switched:
        if speaker == 'A': speaker = 'B'
        else: speaker = 'A'
    file_name = os.path.join(ms_data_dir, split_num, 'sw' + str(file_num) + \
            speaker + '-ms98-a-penn.text')
    df_ms = pd.read_csv(file_name, sep = '\t', names = columns)
    df_ms['ptb_id'] = df_ms.ptb_id.apply(lambda x: x.replace('.', ''))
    df_ms['ms_word'] = df_ms.ms_word.apply(norm_laughter)
    df_ms['ptb_word'] = df_ms.ptb_word.apply(lambda x: x.lower())
    for other_word in OTHER:
        df_ms = df_ms[(df_ms.ms_word != other_word)]
    return df_ms


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
        "//" : [],
        "--" : [],
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

def get_offset_ids(df, colname):
    offsets = [0]
    for i, row in df.iterrows():
        offsets.append(offsets[i] + len(row[colname]) + 1) 
        #+1 accounting for \n
    return offsets

def align_dfs(df_mrg_all, df_comb, file_num, speaker):
    df_mrg = df_mrg_all[(df_mrg_all.speaker==speaker) & \
            (df_mrg_all.file_num==file_num)]
    df_align = df_comb[(df_comb.speaker==speaker) & \
            (df_comb.file_num==file_num)]
    df_mrg['ptb_tokens'] = df_mrg['sentence'].apply(get_tokens)
    df_mrg['mrg'] = df_mrg['mrg'].apply(norm_mrg)
    
    df_align.ptb_tokens = df_align.ptb_tokens.apply(make_list) 
    df_align.ms_tokens = df_align.ms_tokens.apply(make_list) 
    df_align.comb_sentence = df_align.comb_sentence.apply(make_list) 

    this_df2 = df_align.sort_values(by=['turn','sent_num']).reset_index()
    dff_toks = this_df2.ms_tokens.values.copy()
    this_df1 = df_mrg.sort_values(by=['turn','sent_num']).reset_index()
    tree_toks = this_df1.ptb_tokens.values.copy()

    this_df2['offsets'] = get_offset_ids(this_df2, 'ms_tokens')[:-1]
    this_df1['offsets'] = get_offset_ids(this_df1, 'ptb_tokens')[:-1]

    d2 = dict(zip(this_df2['offsets'], zip(this_df2.turn, this_df2.sent_num)))
    d1 = dict(zip(this_df1['offsets'], zip(this_df1.turn, this_df1.sent_num)))
    
    seqa = []
    seqb = []
    for s in dff_toks[:]:
        seqa += s
        seqa += ['\n']
    for s in tree_toks[:]:
        seqb += s
        seqb += ['\n']

    merge, split, other = 0, 0, 0
    all_pairs = []
    all_singles = set([])
    
    # assuming dff_toks have better segmentation:
    # .get_opcodes returns ops to turn a into b 
    sseq = SequenceMatcher(None, seqa, seqb)
    for tag, i1, i2, j1, j2 in sseq.get_opcodes():
        if tag != 'equal':
            if tag == 'delete': 
                if i2-i1 == 1 and seqa[i1:i2][0] == '\n':
                    merge += 1
                elif seqa[i1:i2] == ['--','--'] or seqa[i1:i2] == ['//']:
                    # not an error
                    continue
            elif tag == 'insert' and j2-j1 == 1 and seqb[j1:j2][0] == '\n':
                split += 1
            elif tag == 'replace':
                if seqa[i1:i2] == ['--','--'] and seqb[j1:j2] == ['\n']:
                    split += 1
                if seqa[i1:i2][0] == ''.join(seqb[j1:j2]):
                    # contractions
                    continue
                if seqb[j1:j2][0] == '-'.join(seqa[i1:i2]):
                    # things like eighty-three
                    continue
                else:
                    other += 1
            else:
                other += 1
            seqa_lo0 = set([k for k in d2.keys() if k<=i1])
            seqa_lo1 = set([k for k in d2.keys() if k<=i2])
            dff_sents = seqa_lo1.difference(seqa_lo0)
            dff_sents.add(max(seqa_lo0))
            seqb_lo0 = set([k for k in d1.keys() if k<=j1])
            seqb_lo1 = set([k for k in d1.keys() if k<=j2])
            ptb_sents = seqb_lo1.difference(seqb_lo0)
            ptb_sents.add(max(seqb_lo0))
            ms_sents = sorted([d2[idx] for idx in dff_sents])
            mrg_sents = sorted([d1[idx]for idx in ptb_sents])
            ms_key = make_key(ms_sents)
            mrg_key = make_key(mrg_sents)
            if ms_key == mrg_key: 
                all_singles = all_singles.union(set(ms_sents))
            else: 
                pair_key = ms_key + ':' + mrg_key
                all_pairs.append([ms_sents, mrg_sents])
            # NOTE: could't use dictionary here because of
            # mapping many to many, AND each mapping might reflect 
            # different errors of both sentence boundary differences 
            # and other transcription errors
            #print ("%7s seqa[%d:%d] (%s) seqb[%d:%d] (%s)" % (tag, i1, \
            #        i2, seqa[i1:i2], j1, j2, seqb[j1:j2]))
            #print "\taffected dff sents: ", ms_sents
            #print "\taffected tree sents: ", mrg_sents
            #print
    return all_pairs, all_singles, this_df2, this_df1

def make_key(s_list):
    rep = ''
    i = 0
    for turn, sent_num in s_list:
        if i > 0: name = '_'
        else: name = ''
        name += 't' + str(turn).zfill(3) + 's' + str(sent_num).zfill(3)
        rep = rep + name
        i += 1
    return rep

def process_pairs(all_pairs,df_dff,df_ptb,df_ms,file_num,speaker,out_dir):
    out_name = os.path.join(out_dir, str(file_num) +'_'+ speaker + \
            '_pairs.txt')
    
    fout = open(out_name, 'w')
    df_dff = df_dff.set_index(['turn', 'sent_num'])
    df_ptb = df_ptb.set_index(['turn', 'sent_num'])
    df_ms = df_ms.set_index('uw_tok_id')
    
    ms_sents_set = set([])
    mrg_sents_set = set([])
    for ms_sents, mrg_sents in all_pairs:
        item = str(ms_sents) + "\t" + str(mrg_sents)
        ms_sents_set = ms_sents_set.union(set(ms_sents))
        mrg_sents_set = mrg_sents_set.union(set(mrg_sents))
        print >> fout, item

    print >> fout, "\n"
    print >> fout, "MS sents"
    for turn, sent_num in sorted(ms_sents_set):
        sent_id = "{}_{}{}_{}".format(file_num, speaker, turn, sent_num)
        #print sent_id
        #print df_dff.loc[turn, sent_num].ms_tokens
        ms_sent = ' '.join(clean_up(df_dff.loc[turn, sent_num].ms_tokens))
        ann = df_dff.loc[turn, sent_num].comb_ann
        tok_ids = make_list(df_dff.loc[turn, sent_num].ms_tok_id)
        if not tok_ids: 
            tok_ids = make_list(df_dff.loc[turn, sent_num].ptb_tok_id)
        if not tok_ids: 
            # must be an empty sentence:
            print "Empty sentence: ", sent_id
            continue
        start_tok = '_'.join(tok_ids[0].split('_')[:2])
        end_tok = '_'.join(tok_ids[-1].split('_')[:2])
        start_time = df_ms.loc[start_tok].start_time
        end_time = df_ms.loc[end_tok].end_time
                
        times = "{};{}".format(start_time, end_time)
        item = "{}\t{}\t{}\t{}".format(sent_id, ms_sent, ann, times)
        f_ms = open(os.path.join(out_dir, sent_id + '.ms'), 'w')
        f_ms.write(ms_sent)
        f_ms.close()
        print >> fout, item

    print >> fout, "\n"
    print >> fout, "PTB sents"
    for turn, sent_num in sorted(mrg_sents_set):
        sent_id = "{}_{}{}_{}".format(file_num, speaker, turn, sent_num)
        ptb_sent = ' '.join(df_ptb.loc[turn, sent_num].ptb_tokens)
        ptb_parse = df_ptb.loc[turn, sent_num].mrg
        item = "{}\t{}\t{}".format(sent_id, ptb_sent, ptb_parse)           
        print >> fout, item

    fout.close()


def process_singles(all_singles,df_dff,df_ptb,df_ms,file_num,speaker,out_dir):
    out_name = os.path.join(out_dir, str(file_num) +'_'+ speaker + \
            '_singles.tsv')

    df_dff = df_dff.set_index(['turn', 'sent_num'])
    #print df_dff
    df_ptb = df_ptb.set_index(['turn', 'sent_num'])
    df_ms = df_ms.set_index('uw_tok_id')
    list_row = []

    for turn, sent_num in sorted(all_singles):
        sent_id = "{}_{}{}_{}".format(file_num, speaker, turn, sent_num)
        #print sent_id
        #print df_dff.loc[turn, sent_num].ms_tokens
        ms_sent = ' '.join(clean_up(df_dff.loc[turn, sent_num].ms_tokens))
        ptb_sent = ' '.join(df_ptb.loc[turn, sent_num].ptb_tokens)
        ptb_parse = df_ptb.loc[turn, sent_num].mrg
        ann = df_dff.loc[turn, sent_num].comb_ann
        tok_ids = make_list(df_dff.loc[turn, sent_num].ms_tok_id)
        if not tok_ids: 
            tok_ids = make_list(df_dff.loc[turn, sent_num].ptb_tok_id)
        if not tok_ids: 
            # must be an empty sentence:
            print "Empty sent: ", sent_id
            continue
        start_tok = '_'.join(tok_ids[0].split('_')[:2])
        end_tok = '_'.join(tok_ids[-1].split('_')[:2])
        start_time = df_ms.loc[start_tok].start_time
        end_time = df_ms.loc[end_tok].end_time

        f_ms = open(os.path.join(out_dir, sent_id + '.ms'), 'w')
        f_ms.write(ms_sent)
        f_ms.close()
        list_row.append({'sent_id': sent_id, \
                'ms_sent': ms_sent, \
                'ptb_sent': ptb_sent, \
                'comb_ann': ann, \
                'times': "{};{}".format(start_time, end_time), \
                'mrg': ptb_parse})
    out_df = pd.DataFrame(list_row)
    out_df.to_csv(out_name, sep='\t', index=False, header=True, \
        columns=["sent_id", "ms_sent", "ptb_sent", "comb_ann", "times", "mrg"])


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            'Get sentence-level examples to parse')
    pa.add_argument('--project_dir', \
            default='/g/ssli/projects/disfluencies/switchboard_data', \
            help='project directory')
    pa.add_argument('--ptb_suffix', \
            default='_mrg.tsv', \
            help='ptb mrg file suffix')
    pa.add_argument('--ms_dir_midfix', \
            default='modified_data/alignments_uw_names', \
            help='ms data directory path')
    pa.add_argument('--comb_suffix', \
            default='treebank_msstate_combine_turns_uw_names_tt.tsv', \
            help='sentence boundary combination file suffix')
    pa.add_argument('--file_num', default=2010, type=int, \
            help='file number, 0 for all')

    pa.add_argument('--out_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/prosodic-anomalies', \
            help='output directory')

    args = pa.parse_args()

    project_dir = args.project_dir
    #split = args.split
    out_dir = args.out_dir
    ptb_suffix = args.ptb_suffix
    comb_suffix = args.comb_suffix
    ms_dir_midfix = args.ms_dir_midfix
    file_num = args.file_num

    #df_mrg_all = preprocess_mrg(project_dir, split, ptb_suffix)
    df_mrg_train = preprocess_mrg(project_dir, 'train', ptb_suffix)
    df_mrg_dt = preprocess_mrg(project_dir, 'dev_test', ptb_suffix)
    df_mrg_all = pd.concat([df_mrg_train, df_mrg_dt])
    df_comb = preprocess_comb(project_dir, comb_suffix)    
    mrg_files = set(df_mrg_all.file_num)
    dff_files = set(df_comb.file_num)
    common_files = mrg_files.intersection(dff_files)

    if file_num == 0:
        #for file_num in sorted(common_files):
            # partially processed files case
            #if file_num <= 4617: continue
        for file_num in [4103, 4108, 4171, 4329, 4617]:
            
            for speaker in ['A', 'B']:
                print file_num, speaker
                all_pairs, all_singles, df_dff, df_ptb = align_dfs(df_mrg_all,\
                        df_comb, file_num, speaker)
                df_ms = preprocess_ms(project_dir, ms_dir_midfix, file_num, \
                        speaker)
                if not all_singles:
                    print "No single-sent errors!", file_num, speaker
                else:
                    process_singles(all_singles, df_dff, df_ptb, df_ms, \
                            file_num, speaker, out_dir)
                if not all_pairs:
                    print "No pair-sent errors!", file_num, speaker
                else:
                    process_pairs(all_pairs, df_dff, df_ptb, df_ms, file_num, \
                            speaker, out_dir)
    
    else:
        for speaker in ['A', 'B']:
            all_pairs, all_singles, df_dff, df_ptb = align_dfs(df_mrg_all, \
                    df_comb, file_num, speaker)
            df_ms = preprocess_ms(project_dir, ms_dir_midfix, file_num, \
                    speaker)
            #for s in sorted(all_pairs): print s
            #for s in sorted(all_singles): print s
            if not all_singles:
                print "No single-sent errors!", file_num, speaker
            else:
                process_singles(all_singles, df_dff, df_ptb, df_ms, file_num, \
                        speaker, out_dir)
            if not all_pairs:
                print "No pair-sent errors!", file_num, speaker
            else:
                process_pairs(all_pairs, df_dff, df_ptb, df_ms, file_num, \
                        speaker, out_dir)

    
