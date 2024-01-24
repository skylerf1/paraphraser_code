# Courtney Napoles
# <napoles@cs.jhu.edu>
# 21 June 2015
# ##
# gleu.py
# 
# This script calculates the GLEU score of a sentence, as described in
# our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
# 
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# Updated 6 9 2017: Fixed inverse brevity penalty
# 
# This script was adapted from bleu.py by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import math
from collections import Counter
import sys
import os
import scipy.stats
import numpy as np
import random
import nltk

class GLEU :

    def __init__(self,n=4) :
        self.order = n

    def load_hypothesis_sentence(self,hypothesis) :
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [ self.get_ngram_counts(hypothesis,n)
                               for n in range(1,self.order+1) ]

    def load_sources(self,sources) :
        self.all_s_ngrams = [ [self.get_ngram_counts(line.split(),n)
                                for n in range(1,self.order+1) ]
                              for line in sources]

    def load_references(self,references) :
        self.refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for rpath in references:
            for i,line in enumerate(rpath.split('\n')):
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))
        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [Counter() for i in range(self.order) ]

        self.all_r_ngrams = [ ]
        for refset in self.refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys() :
                    self.all_rngrams_freq[n-1][k]+=1

                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self,sentence,n) :
        return Counter([tuple(sentence[i:i+n])
                        for i in range(len(sentence)+1-n)])

    # returns ngrams in a but not in b
    def get_ngram_diff(self,a,b) :
        diff = Counter(a)
        for k in (set(a) & set(b)) :
            del diff[k]
        return diff

    def normalization(self,ngram,n) :
        return 1.0*self.all_rngrams_freq[n-1][ngram]/len(self.rlens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self,i,r_ind=None):

      hlen = self.hlen
      rlen = self.rlens[i][r_ind]
      
      yield hlen
      yield rlen
      for n in range(1,self.order+1):
        h_ngrams = self.this_h_ngrams[n-1]
        #print('h_ngrams:', h_ngrams)
        s_ngrams = self.all_s_ngrams[i][n-1]
        #print('s_ngrams:', s_ngrams)
        r_ngrams = self.get_ngram_counts(self.refs[i][r_ind],n)
        #print('r_ngramms)

        s_ngram_diff = self.get_ngram_diff(s_ngrams,r_ngrams)
        #print('s_ngrams_diff:', s_ngram_diff)
        #print('first yield:', max([ sum( (h_ngrams & r_ngrams).values() ) - sum( (h_ngrams & s_ngram_diff).values() ), 0 ]))
        #print('second yield:', max([hlen+1-n, 0]))
        #print('first part:', sum((h_ngrams & r_ngrams).values()))
        #print('second part:', sum( (h_ngrams & s_ngram_diff).values() ))
        yield max([ sum( (h_ngrams & r_ngrams).values() ) - \
                    sum( (h_ngrams & s_ngram_diff).values() ), 0 ])

        yield max([hlen+1-n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self,stats,smooth=False):
        # smooth 0 counts for sentence-level scores
        #print(smooth)
        if smooth:
            stats = [ s if s != 0 else 1 for s in stats ]
        if len(list(filter(lambda x: x==0, stats))) > 0:
            return 0
        #print('c, r, numerator1, denominator1', stats)
        #print('stats 2::2', stats[2::2], 'stats3::2', stats[3::2])
        (c, r) = stats[:2]
        #print('c', c)
        #print('r', r)
        log_gleu_prec = sum([math.log(float(x)/y)
                             for x,y in zip(stats[2::2],stats[3::2])]) / 4.0
        #print('log gleu prec:', log_gleu_prec)
        #print('math_exp:', math.exp(min([0, 1-float(r)/c]) + log_gleu_prec))
        return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)

def get_gleu_stats(scores) :
    mean = np.mean(scores)
    std = np.std(scores)
    ci = scipy.stats.norm.interval(0.95,loc=mean,scale=std)
    return ['%f'%mean,
            '%f'%std,
            '(%.3f,%.3f)'%(ci[0],ci[1])]
def prepare_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return ' '.join(tokens)

def gleu_calculator_mine(sources, candidate, references, n, num_iterations):
    if len(references) == 1:
        num_iterations = 1
    source = [prepare_text(so) for so in [sources]]
    references = [prepare_text(refer) for refer in references]
    hypothesis = [prepare_text(candidate)]

    #print('reference:', references)
    #print('hypothesis', hypothesis)
    #print('source', source)
    
    gleu_calculator = GLEU(n=n)
    gleu_calculator.load_sources(source)
    gleu_calculator.load_references(references)
    
    for hpath in hypothesis:
        instream = sys.stdin if hpath == '-' else [hpath]
        hyp = [line.split() for line in instream]
        #print(hyp)

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j*101)
            indices.append([random.randint(0,len(references)-1)
                            for i in range(len(hyp))])

        iter_stats = [[0 for i in range(2*n+2)]
                        for j in range(num_iterations)]
       
        for i,h in enumerate(hyp) :
            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(len(references)) ]
            #print('stats by ref', stats_by_ref)
            for j in range(num_iterations) :
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [ s for s in gleu_calculator.gleu_stats(
                        i,r_ind=ref) ]
                    stats_by_ref[ref] = this_stats
                    #print('if not None stats by ref', stats_by_ref[ref])
                iter_stats[j] = [ sum(scores)
                                    for scores in zip(iter_stats[j], this_stats)]

            return(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True)
                                    for stats in iter_stats]))
def compute_gleu(predictions, references, sources, n=4, num_iterations=500):
    value = 0
    for i in range(len(predictions)):
        value += float(gleu_calculator_mine(sources[i], predictions[i], references[i], n=n, num_iterations=num_iterations))
    return {'GLEU+' : round(value/len(predictions), 3)}
