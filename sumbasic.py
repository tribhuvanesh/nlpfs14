import argparse
from collections import defaultdict
import json
from corenlp.corenlp import ProcessError

import nlpio
from lxml import etree
import nltk

def main():
    parser = argparse.ArgumentParser(description='Create a frequency dictionary of lemmatized words')
    parser.add_argument('file_list', metavar='INFILE', type=str, help='File containing paths to documents')
    parser.add_argument('freq_dict', metavar='OUTFILE', type=str, help='Name of output json file')
    args = parser.parse_args()

    freqdct = defaultdict(int)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()

    with open(args.file_list, 'r') as doclist:
        for line in doclist:
            docfile = line.strip()
            print 'Processing %s' % docfile
            with open(docfile, 'r') as doctext:
                doc = etree.parse(doctext)
                text = doc.find('.//TEXT').text

                # Some basic preprocessing on text
                text.replace('\n', '')
                text = text.lower().strip()

                # try:
                #     parseout = nlpio.stanfordParse(text)
                #     for sent in parseout['sentences']:
                #         for word in sent['words']:
                #             freqdct[ word[1]['Lemma'] ] += 1
                # except ProcessError:
                #     print 'Could not process line: ', text

                for sent in sent_tokenizer.tokenize(text):
                    for word in tokenizer.tokenize(sent):
                        freqdct[word] += 1


    # Dump freqdct to file
    with open(args.freq_dict, 'w') as json_file:
        json_str = json.dumps(freqdct)
        json_file.write(json_str.replace(' ', ''))


if __name__ == '__main__':
    main()
