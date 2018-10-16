from nltk import word_tokenize, pos_tag, ne_chunk
import pandas as pd
import numpy as np
import nltk


checked_tags = ['PER', 'LOC', 'ORG']
checked_tags_stanford = ['PERSON', 'ORGANIZATION', 'LOCATION']
df_columns = ['WikiGold Token','WikiGold Tag', 'Predicted Token', 'Predicted Tag', 'Condition']


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


def replace_stanford_ent_type(ent_type_):
    if 'PER' in ent_type_:
        return 'PER'
    elif 'LOC' in ent_type_:
        return 'LOC'
    elif 'ORG' in ent_type_ or 'GPE' in ent_type_:
        return 'ORG'
    return np.nan


def compare_tokens(token, tagged_word, dataf):
    if token[1] in checked_tags and tagged_word[1] in checked_tags_stanford: # if same tags, then TP, else, FP
        if token[1] == replace_stanford_ent_type(tagged_word[1]):
            condition = 'TP'
        else:
            condition = 'FP'

    elif token[1] in checked_tags and tagged_word[1] not in checked_tags_stanford: # FP
        condition = 'FP'

    elif token[1] not in checked_tags and tagged_word[1] not in checked_tags_stanford: # TN
        condition = 'TN'

    elif token[1] not in checked_tags and tagged_word[1] in checked_tags_stanford: # FN
        condition = 'FN'

    if condition:
        dataf = dataf.append(dict(zip(df_columns,
                               [token[0],
                               token[1],
                               tagged_word[0],
                               replace_stanford_ent_type(tagged_word[1]),
                               condition])),
                        ignore_index=True)

    else: # not found, FP
        if token[1] in checked_tags:
            dataf = dataf.append(dict(zip(df_columns,
                       [token[0], token[1], tagged_word[0], np.nan, 'FP'])), ignore_index=True)
        # elif token[1] not in checked_tags:
        print("***************** not found ", tagged_word)
    return dataf


def main():

    jar = 'stanford-ner.jar'
    model = 'english.all.3class.distsim.crf.ser.gz'


    df = pd.DataFrame(columns=df_columns)

    raw = open('wikigold.conll.txt', 'r')

    sentences_for_spacy = ''
    wiki_sentence = []
    num_lines_skipped = 0


    for s in raw.readlines():
        if s == "\n" or 'DOCSTART' in s:
            num_lines_skipped += 1
        if s != "\n" and 'DOCSTART' not in s:
            word_and_tag = replace_wiki_tags(s.split())
            wiki_sentence.append(word_and_tag)
            sentences_for_spacy += s.split()[0] + ' '
        elif 'DOCSTART' in s:
            tagged_doc = ne_chunk(pos_tag(word_tokenize(sentences_for_spacy)))
            tagged_words = list(tagged_doc.leaves())
            if len(tagged_words) != len(wiki_sentence):
                a = 0
                for node in tagged_doc:
                    if type(node) is nltk.Tree:
                        for l in node.leaves():
                            _ = wiki_sentence[a]
                            _ = l[0]
                            df = compare_tokens(wiki_sentence[a], (l[0], node.label()), df)
                    else:
                        df = compare_tokens(wiki_sentence[a], node, df)
                    if node[0] != wiki_sentence[a][0] and node == ('.', '.') and '.' in wiki_sentence[a-1][0]:
                            a -= 1
                    a += 1
            else:
                i = 0
                for node in tagged_doc:
                    if type(node) is nltk.Tree:
                        for l in node.leaves():
                            _ = wiki_sentence[i]
                            _ = l[0]
                            df = compare_tokens(wiki_sentence[i], (l[0], node.label()), df)
                            i+=1
                    else:
                        df = compare_tokens(wiki_sentence[i], node, df)
                        i+=1

            sentences_for_spacy = ''
            wiki_sentence = []

    # df['Correct?'] = df.apply(lambda x: x['WikiGold Tag'] is x['spaCy Tag'], axis=1)

    # print(df)

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