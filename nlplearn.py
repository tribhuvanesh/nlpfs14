from nlpio import *

import numpy as np
import logging
import random
import pprint

from sklearn.base import BaseEstimator,TransformerMixin

# NLTK stuff
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

class RougeScorer(object):
    #variant can be any rouge variant available in the output
    #metric should be one of P,R,F
    def __init__(self,variant='ROUGE-1',metric='F'):
        self.variant = variant
        self.metric = metric
        self.pp = pprint.PrettyPrinter()

    def __call__(self,estimator,documents,predictions=None):
        if predictions is None:
            predictions = estimator.predict(documents)
        results = evaluateRouge(documents,predictions)
        self.pp.pprint(results)
        return results[self.variant][self.metric][0]


class SimpleTextCleaner(BaseEstimator,TransformerMixin):
    '''
    Step 1 - Clean text
    '''
    #TODO: make better
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        # for doc in documents:
        #     doc.text = re.sub("`|'|\"","",doc.text)
        #     doc.text = re.sub("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.","\\1",doc.text)
        return documents


class SentenceSplitter(BaseEstimator,TransformerMixin):
    '''
    Step 2 - Split sentences
    '''
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        sentence_splitter = PunktSentenceTokenizer()
        for doc in documents:
            if not 'sentences' in doc.ext:
                doc.ext['sentences'] = [s.strip() for s in sentence_splitter.tokenize(doc.text)]
        # for doc in documents:
        #     if not 'sentences' in doc.ext:
        #         doc.ext['sentences'] = [s.strip() for s in doc.text.split('.') if s]
        return documents


class HeadlineEstimator(BaseEstimator):
    '''Estimates, for a given document, its headline'''
    def __init__(self):
        pass

    def fit(self, documents, y=None): #we generate the headlines directly from the documents, so we don't need a "y"
        return self

    def predict(self, documents):
        return [doc.ext['sentences'][0] for doc in documents] #for now this just predicts the first sentence

