
# coding: utf-8

# In[1]:


# %load debug
import sys
import render_tree, init, treebanks, parse_errors, head_finder, tree_transform
import pstree
from collections import defaultdict
from StringIO import StringIO
from difflib import SequenceMatcher
from nltk.tree import Tree


# In[12]:


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
    new_parent = left_sibling.parent
    node_to_add.parent = new_parent
    idx = new_parent.subtrees.index(left_sibling)
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

# In[19]:


#ms_file = '../samples/4936_A9_0.ms.bkout'
ms_file = '../samples/2015_B30_2.ms.bkout'
ms_tree_candidates = open(ms_file).readlines()
ms_tree_candidates = [x.strip() for x in ms_tree_candidates]
ms_tree_candidates = ["(ROOT"+ x[1:] for x in ms_tree_candidates]

#ptb_tree_mrg = "(ROOT (S (INTJ (UH Uh) ) (INTJ (UH uh) ) (NP-SBJ-1 (PRP$ my) (NN wife) ) (VP (VBZ has) (VP (VBN picked) (PRT (RB up) ) (NP (NP (DT a) (NN couple) ) (PP (IN of) (NP (NNS things) ) ) ) (S-ADV (VP (VBG saying) (INTJ (UH uh) ) (S-SEZ (INTJ (UH boy) ) (SBAR-ADV (IN if) (S (NP-SBJ (PRP we) ) (VP (MD could) (VP (VB refinish) (NP (DT that) ) ) ) ) ) (NP-SBJ (DT that) ) (VP (MD would) (VP (VB be) (NP-PRD (NP (DT a) (JJ beautiful) (NN piece) ) (PP (IN of) (NP (NN furniture) ) ) ) ) ) ) ) ) ) ) ) )"
ptb_tree_mrg = "(ROOT (S (NP-SBJ (PRP you) ) (VP (VP (VBD started) (PRT (RP off) ) ) (CC and) (VP (VBD said) ) ) ) )"

temp = pstree.tree_from_text(ms_tree_candidates[0])
ms_words = temp.word_yield(as_list=True)
ptb_tree = pstree.tree_from_text(ptb_tree_mrg)
treebanks.remove_function_tags(ptb_tree)
ptb_words = ptb_tree.word_yield(as_list=True)
ptb_words = [x.lower() for x in ptb_words]


# In[20]:


t1 = Tree.fromstring(ptb_tree_mrg)
t1.pretty_print()

t2 = Tree.fromstring(ms_tree_candidates[0])
t2.pretty_print()


# In[21]:


sseq = SequenceMatcher(None, ptb_words, ms_words)
# .get_opcodes returns ops to turn a into b 
ops = []
for info in sseq.get_opcodes():
    if info[0] != 'equal': ops.append(info)

print ops


# In[22]:

for op in ops[::-1]:
    tag, i1, i2, j1, j2 = op
    print tag
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
        insert_words(ptb_tree, ms_words, i1, i2, j1, j2)
    ptb_tree.check_consistency()

s_new = str(ptb_tree)
t_new = Tree.fromstring(s_new)
t_new.pretty_print()

#l = ptb_tree.get_nodes(end=3)
#r = ptb_tree.get_nodes(start=3)
#
#print
#foo = ptb_tree.get_nodes(end=3, request='highest')
#print foo.label, foo.span
#
#print
#foo = ptb_tree.get_nodes(end=3, request='lowest')
#print foo.label, foo.span


# In[26]:


#import transform_search
#from classify_english import classify
#init_errors = parse_errors.get_errors(ptb_tree, temp)
#iters, path = transform_search.greedy_search(temp, ptb_tree, classify)
#
#for e in init_errors: print e
#print
#for p in path: print p

