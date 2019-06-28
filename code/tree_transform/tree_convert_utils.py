#!/usr/bin/env python
from parse_analyzer import render_tree, init, treebanks, parse_errors, nlp_eval
from parse_analyzer import pstree, transform_search, head_finder, tree_transform
import sys, argparse, os
from parse_analyzer.classify_english import classify
from collections import defaultdict
from StringIO import StringIO
from difflib import SequenceMatcher
from nltk.tree import Tree
from nltk.draw.tree import TreeView
import cPickle as pickle

# dictionary of most common tags
tag_counts = pickle.load(open('tag_counts.pickle'))

# replace word by word 
def replace_words(ptb_tree, ms_words, i1, i2, j1, j2):
    k = 0
    for sidx in range(i1, i2):
        this_node = ptb_tree.get_nodes(start=sidx, end=sidx+1)
        leaf = this_node[-1]
        assert leaf.is_terminal()
        leaf.word = ms_words[j1+k]
        k += 1
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

def look_up_tag(word):
    if word not in tag_counts: 
        return 'XX'
    tag = tag_counts[word].most_common(1)[0][0]
    return tag
    
def insert_words(ptb_tree, ms_words, i1, i2, j1, j2):
    to_insert = ms_words[j1:j2]
    str_mrg = ['(' + look_up_tag(w) + ' ' + w +')' for w in to_insert]
    mrg = '(X ' + ' '.join(str_mrg) + ' )'
    node_to_add = pstree.tree_from_text(mrg)
    
    # i1 = i2 in this case; check if insert at end points
    # if inserting at endpoints, do merge 
    if i1 == 0 or i1 == len(ptb_tree.word_yield(as_list=True)):
        if i1 == 0:
            ptb_tree = merge_trees(node_to_add, ptb_tree)
        else:
            ptb_tree = merge_trees(ptb_tree, node_to_add)
        return ptb_tree

    # else: inserting in the middle of the sentence
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
    return ptb_tree

def merge_trees(t1, t2):
    left = treebanks.homogenise_tree(t1)
    right = treebanks.homogenise_tree(t2)
    sl, el = left.span
    sr, er = right.span
    # take the highest node under ROOT, usually "S" that has same span
    right_sub_nodes = right.get_nodes(start=sr, end=er)
    # print right_sub_nodes
    right_sub = right.get_nodes(start=sr, end=er)[-1]
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

    sseq = SequenceMatcher(None, ptb_words, ms_words)
    # .get_opcodes returns ops to turn a into b 
    ops = []
    for info in sseq.get_opcodes():
        if info[0] != 'equal': ops.append(info)
    
    # Fix word-level mismatch first
    for op in ops[::-1]:
        tag, i1, i2, j1, j2 = op
        #print op, ptb_words[i1:i2], ms_words[j1:j2] 
        if tag == 'replace':
            # easy case: same number of words --> just substitute
            if (i2 - i1) == (j2 - j1):
                replace_words(ptb_tree, ms_words, i1, i2, j1, j2)
            # harder case: replace ptb_words[i1:i2] by ms_words[j1:j2]
            else:
                delete_words(ptb_tree, i1, i2)
                ptb_tree = insert_words(ptb_tree, ms_words, i1, i2, j1, j2)
        elif tag == 'delete':
            delete_words(ptb_tree, i1, i2)
        else:
            ptb_tree = insert_words(ptb_tree, ms_words, i1, i2, j1, j2)
        ptb_tree.check_consistency()

    # Now look at structure
    # initial "errors" in ms_tree compared to "gold" of ptb_tree
    # counts_for_prf(test, gold)
    # get_errors(test, gold)
    # transform_search(gold, test)
    # find transformation from ms_tree to ptb_tree
    if rev:
        init_errors = parse_errors.get_errors(ptb_tree, ms_tree)
        iters, path = transform_search.greedy_search(ms_tree,ptb_tree,classify)
        m12, g12, t12, _, _ = parse_errors.counts_for_prf(ptb_tree, ms_tree)
        p12, r12, f12 = nlp_eval.calc_prf(m12, g12, t12)
    else:
        init_errors = parse_errors.get_errors(ms_tree, ptb_tree)
        iters, path = transform_search.greedy_search(ptb_tree,ms_tree,classify)
        m12, g12, t12, _, _ = parse_errors.counts_for_prf(ms_tree, ptb_tree)
        p12, r12, f12 = nlp_eval.calc_prf(m12, g12, t12)

    # count POS tag differences:
    ptb_leaves = [x for x in ptb_tree.get_nodes() if x.is_terminal()]
    ptb_tags = [x.label for x in ptb_leaves]
    ms_leaves = [x for x in ms_tree.get_nodes() if x.is_terminal()]
    ms_tags = [x.label for x in ms_leaves]
    matched_tags = sum([x==y for x,y in zip(ptb_tags, ms_tags)])
    
    return init_errors, iters, path, ptb_tree, f12, matched_tags

# Produce intermediate trees from PTB versions, updated according to MS words
def align_pairs(file_num, speaker, ms_hypo_dir, draw_tree):
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
        ms_side = set([item for sublist in ms_side for item in sublist])
        ptb_side = [x[1] for x in pairs if (turn, sent_num) in x[0]]
        ptb_side = set([item for sublist in ptb_side for item in sublist])
        idx = is_in_set(ms_side, sides)
        if idx > -1:
            ptb_side = ptb_side.union(sides[idx][1])
            sides[idx] = [ms_side, ptb_side]
        else:
            sides.append([ms_side, ptb_side])
    
    for ms_side, ptb_side in sides:
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
                        delete_words(this_tree, i1, i2)
                        this_tree = insert_words(this_tree, ms_tokens, i1, i2, \
                                j1, j2)
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
    return temp_mrg

