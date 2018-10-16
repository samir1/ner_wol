import spacy
from pprint import pprint
import pandas as pd
import numpy as np



checked_spacy_tags = ['PERSON', 'LOC', 'ORG', 'GPE']
checked_tags = ['PER', 'LOC', 'ORG']
df_columns = ['WikiGold Token','WikiGold Tag', 'Predicted Token', 'Predicted Tag', 'Condition']

# In the evaluation, we ignore the MISC type
# and map the gold standard types:
# PER, LOC and ORG to normalized types PERSON,
# LOCATION and ORGANIZATION respectively.

# The related output
# tags include “PERSON”, “LOC”, “ORG”, “GPE” etc.
# For spaCy outputs, we map
# "PERSON" to "PER"
# “LOC” to LOC,
# “ORG” and “GPE” to "ORG"
# and ignore all other types.

def replace_wiki_tags(word_and_tag):
    if 'PER' in word_and_tag[1]:
        word_and_tag[1] = 'PER'
    elif 'LOC' in word_and_tag[1]:
        word_and_tag[1] = 'LOC'
    elif 'ORG' in word_and_tag[1]:
        word_and_tag[1] = 'ORG'
    else:
        word_and_tag[1] = np.nan
    return word_and_tag

def replace_spacy_ent_type(ent_type_):
        if 'PER' in ent_type_:
            return 'PER'
        elif 'LOC' in ent_type_:
            return 'LOC'
        elif 'ORG' in ent_type_ or 'GPE' in ent_type_:
            return 'ORG'
        return np.nan

def main():

    nlp = spacy.load('en')
    # nlp = spacy.load('en_core_web_lg')

    df = pd.DataFrame(columns=df_columns)

    raw = open('wikigold.conll.txt', 'r')

    sentences_for_spacy = ''
    wiki_sentence = []
    # docs = []
    num_lines_skipped = 0
    for i,s in enumerate(raw.readlines()):
        if s == "\n" or 'DOCSTART' in s:
            num_lines_skipped += 1
        if s != "\n" and 'DOCSTART' not in s:
            word_and_tag = replace_wiki_tags(s.split())
            wiki_sentence.append(word_and_tag)
            sentences_for_spacy += s.split()[0] + ' '
        elif 'DOCSTART' in s:
            # break

            # print(wiki_sentence)
            # print(sentences_for_spacy)

            doc = nlp(sentences_for_spacy)
            # print(doc.ents)

            start = 0
            for token in wiki_sentence:
                # start = doc.text.index(token[0]+' ', start)
                start = doc.text.index(' '+token[0]+' ')+1 if ' '+token[0]+' ' in doc.text else doc.text.index(token[0]+' ', start)
                end = start + len(token[0])
                # if token[1] in checked_tags:
                if doc.char_span(start,end) and doc.char_span(start,end).root:
                    # print([token[0], token[1], doc.char_span(start,end).root.text, replace_spacy_ent_type(doc.char_span(start,end).root.ent_type_)])
                    if token[1] in checked_tags and doc.char_span(start,end).root.ent_type_ in checked_spacy_tags: # if same tags, then TP, else, FP
                        if token[1] == replace_spacy_ent_type(doc.char_span(start,end).root.ent_type_):
                            condition = 'TP'
                        else:
                            condition = 'FP'

                    elif token[1] in checked_tags and doc.char_span(start,end).root.ent_type_ not in checked_spacy_tags: # FP
                        condition = 'FP'

                    elif token[1] not in checked_tags and doc.char_span(start,end).root.ent_type_ not in checked_spacy_tags: # TN
                        condition = 'TN'

                    elif token[1] not in checked_tags and doc.char_span(start,end).root.ent_type_ in checked_spacy_tags: # FN
                        condition = 'FN'

                    if condition:
                        df = df.append(dict(zip(df_columns,
                                               [token[0],
                                               token[1],
                                               doc.char_span(start,end).root.text,
                                               replace_spacy_ent_type(doc.char_span(start,end).root.ent_type_),
                                               condition])),
                                        ignore_index=True)

                else: # not found, FP
                    if token[1] in checked_tags:
                        df = df.append(dict(zip(df_columns,
                                   [token[0], token[1], sentences_for_spacy[start:end], np.nan, 'FP'])), ignore_index=True)
                    # elif token[1] not in checked_tags:
                    print("***************** not found ",
                          token, doc.text[start-30:start]+'|'+doc.text[start:end]+'|'+doc.text[end:end+30])

            sentences_for_spacy = ''
            wiki_sentence = []

    # df['Correct?'] = df.apply(lambda x: x['WikiGold Tag'] is x['spaCy Tag'], axis=1)

    # print(df)

    # print(np.sum(df['Correct?'])/len(df['Correct?']))
    print('Num lines skipped', num_lines_skipped)
    print()
    condition_vals = df['Condition'].value_counts()
    print(condition_vals)
    precision = condition_vals['TP']/(condition_vals['TP'] + condition_vals['FP'])
    print('Precision', precision)
    recall = condition_vals['TP']/(condition_vals['TP'] + condition_vals['FN'])
    print('Recall', recall)
    fscore = 2*((precision*recall)/(precision+recall))
    print('F1 score', fscore)

    return df

if __name__ == "__main__":
    main()