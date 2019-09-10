## Paper
This is code used in the paper [On the Role of Style in Parsing Speech with Neural Models](https://ttmt001.github.io/pdfs/3122_Paper.pdf)

## Usage
example_job.sh contains an example of how to run the code (training and evaluation). Note that this assumes the parse trees and speech features are already available. Parse trees should follow the Penn Treebank format; acoustic-prosodic features need to be extracted as described in the Parsing Speech paper. We used Kaldi; although these can't be used out-of-the box, example codes for the feature extraction pipeline is in https://github.com/trangham283/prosody_nlp/tree/master/code/kaldi_scripts and https://github.com/trangham283/prosody_nlp/tree/master/code/feature_extraction

## TODO
Add example data

## Acknowledgements
The code in this repository is based on the implementations in these papers:
1. "Constituency Parsing with a Self-Attentive Encoder", Kitaev and Klein, ACL 2018: 
https://github.com/nikitakit/self-attentive-parser 
2. "Parsing Speech: A Neural Approach to Integrating Lexical and Acoustic-Prosodic Information", Tran et al., NAACL 2018: 
https://github.com/shtoshni92/speech_parsing


## Modifications from original code cited above
* Working with Pytorch 0.4.x instead of 0.3.x or tensorflow 
* Incorporating speech features (CNN module) 
