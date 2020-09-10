# Python 2; Python 3 should work except the print statements
import re
import os
import pandas
from glob import glob
from tree_utils import delete_constituents, linearize_tree, detach_brackets

# Not used but left here for reference of train/dev splits
# Based on Honnibal's convert.py
# https://github.com/syllog1sm/swbd_tools
def divide_files(swbd_loc):
    """Divide data into train/dev/test/dev2 split following Johnson and Charniak
    division"""
    swbd_loc = os.path.join(swbd_loc, 'parsed', 'mrg', 'swbd')
    train = []
    test = []
    dev = []
    dev2 = []
    files = glob(swbd_loc+'/2/*') + glob(swbd_loc+'/3/*') + glob(swbd_loc+'/4/*')
 
    for filename in files:
        if not filename.endswith('.mrg'): continue
        filenum = int(filename[-8:-4])
        if filenum < 4000:
            train.append(filename)
        elif 4000 < filenum <= 4153: # fixed file offset: test = 4004-4153
            test.append(filename)
        elif 4500 < filenum <= 4936:
            dev.append(filename)
        else:
            dev2.append(filename)
    return train, test, dev, dev2


def make_csv(ptb_loc, split, out_dir):
    swbd_loc = os.path.join(ptb_loc, 'parsed', 'mrg', 'swbd')
    if split == 'train':
        files = glob(swbd_loc+'/2/*.mrg') + glob(swbd_loc+'/3/*.mrg')
    else:
        files = glob(swbd_loc+'/4/*.mrg')
    list_row = []
    for f in files:
        fname = f[-10:-4]
        mrg_txt = open(f).read()
        turns = mrg_txt.split('( (CODE')[1:]
        for turn in turns:
            raw_lines = turn.split('\n')
            # first line gives speaker info
            unit_label = raw_lines[0].split()[1].rstrip(')').lstrip('Speaker')
            new_tree = []
            tree_id = 0
            for i in range(1,len(raw_lines)-1):
                line = raw_lines[i]
                #print line
                next_line = raw_lines[i+1]
                if not line:
                    continue
                if next_line and next_line[0] != '(':
                    new_tree.append(line.strip())
                else:
                    new_tree.append(line.strip())
                    if new_tree:
                        print fname, unit_label, tree_id
                        #print new_tree
                        temp_tree = ' '.join(new_tree).split()
                        temp_tree = detach_brackets(temp_tree)
                        temp_tree = ' '.join(temp_tree).split()
                        new_tree = delete_constituents(temp_tree)
                        sent, parse = linearize_tree(new_tree)
                        #print ' '.join(new_tree)
                        list_row.append({'mrg': ' '.join(new_tree),\
                                'sentence': ' '.join(sent), \
                                'parse': ' '.join(parse), \
                                'file_id': fname,\
                                'sent_id': unit_label, \
                                'tree_id': tree_id})
                        tree_id += 1
                        new_tree = []

    data_df = pandas.DataFrame(list_row)
    outfile = os.path.join(out_dir, split + '_mrg.tsv')
    data_df.to_csv(outfile, sep='\t',index=False)


if __name__ == '__main__':
    ptb_loc = '/g/ssli/data/treebank/release3'
    out_dir = '/s0/ttmt001/speech_parsing'
    split = 'train'
    #split = 'dev_test'
    # dev_test were combined
    make_csv(ptb_loc, split, out_dir)



