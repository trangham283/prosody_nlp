#!/usr/bin/env python
import sys, argparse, os, glob
import cPickle as pickle
import pandas as pd
import numpy as np

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

def sort_keys(my_list):
    new_list = []
    for a in my_list:
        turn = int(a.split('_')[1][1:])
        sent_num = int(a.split('_')[2])
        new_list.append([a, (turn, sent_num)])
    sorted_keys = sorted(new_list, key=lambda x: x[1])
    sorted_keys = [x[0] for x in sorted_keys]
    return sorted_keys

ERR = ['INS', 'DEL', 'SUB']
def get_stats(in_dir, file_num, speaker, ftype, out_dir):
    if ftype == 'singles':
        f = os.path.join(in_dir, "{}_{}_singles.tsv".format(file_num, speaker))
    else:
        f = os.path.join(in_dir, "{}_{}_updated_pairs.tsv".format(file_num, \
                speaker))
    df = pd.read_csv(f, sep="\t")
    df['ann'] = df.comb_ann.apply(make_list)
    df['raw_len'] = df.ann.apply(len)
    df['err_sub'] = df.comb_ann.apply(lambda x: x.count('SUB'))
    df['err_del'] = df.comb_ann.apply(lambda x: x.count('DEL'))
    df['err_ins'] = df.comb_ann.apply(lambda x: x.count('INS'))
    df['ftype'] = ftype
    out_name = os.path.join(out_dir, "{}_{}_{}_err_counts.tsv".format(file_num,\
            speaker, ftype))
    to_keep = ['sent_id', 'raw_len', 'err_sub', 'err_ins', 'err_del', 'ftype']
    df_new = df[to_keep]
    #df_new.to_csv(out_name,sep="\t",header=True,columns=to_keep,index=False)
    return df_new

def concat_stats(all_files, out_dir, ms_hypo_dir):
    list_df = []
    for f in all_files:
        fname = os.path.basename(f).split('_')
        file_num = int(fname[0])
        speaker = fname[1]
        ftype = fname[-1].split('.')[0]
        print file_num, speaker, ftype
        df = get_stats(ms_hypo_dir, file_num, speaker, ftype, out_dir)
        list_df.append(df)
    all_df = pd.concat(list_df)
    out_name = os.path.join(out_dir, 'align_stats.tsv')
    all_df.to_csv(out_name, sep="\t", index=False)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
            "Count errors/various sentence stats to get examples")
    pa.add_argument('--ms_hypo_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_info',\
            help='directory of ms hypotheses')
    pa.add_argument('--file_num', default=2005, type=int, \
            help='file number, 0 for all')
    pa.add_argument('--speaker', default='A', type=str, \
            help='speaker')
    pa.add_argument('--out_dir', type=str, \
            default='/s0/ttmt001/speech_parsing/ms_ptb_align_stats', \
            help='output directory')

    # previously both ms_hypo_dir and out_dir were "samples"
    args = pa.parse_args()

    file_num = args.file_num
    speaker = args.speaker
    out_dir = args.out_dir
    ms_hypo_dir = args.ms_hypo_dir

    all_files = glob.glob(ms_hypo_dir + "/*_singles.tsv")
    all_files += glob.glob(ms_hypo_dir + "/*_updated_pairs.tsv")

    # get all stats:
    # concat_stats(all_files, out_dir, ms_hypo_dir)

    # analysis
    out_name = os.path.join(out_dir, 'align_stats.tsv')
    all_df = pd.read_csv(out_name, sep='\t')
    all_df['total_err'] = all_df['err_sub'] + all_df['err_del'] + \
            all_df['err_ins']
    mask1 = all_df.ftype == 'singles'
    mask2 = all_df.ftype == 'pairs'

    df1 = all_df[mask1]
    df2 = all_df[mask2]
    df1_sorted = df1.sort_values(by=['total_err', 'raw_len']).tail(30)
    print "From singles sentences"
    print df1_sorted
    for k in sorted(df1_sorted.sent_id.values): print k
    print

    df2_sorted = df2.sort_values(by=['total_err', 'raw_len']).tail(30)
    print "From pairs sentences"
    print df2_sorted
    for k in sorted(df2_sorted.sent_id.values): print k
    

