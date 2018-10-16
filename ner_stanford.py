from nltk.tag import StanfordNERTagger
import pandas as pd
import numpy as np




df_columns = ['WikiGold Token','WikiGold Tag', 'Predicted Token', 'Predicted Tag', 'Condition']
checked_tags = ['PER', 'LOC', 'ORG']
checked_tags_stanford = ['PERSON', 'ORGANIZATION', 'LOCATION']


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

def main():
    jar = 'stanford-ner.jar'
    model = 'english.all.3class.distsim.crf.ser.gz'

    st = StanfordNERTagger(model, jar, encoding='utf-8')
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

            tagged_doc = st.tag(sentences_for_spacy.split())
            # print(tagged_doc)
            # print(len(tagged_doc))
            # print('------------------------------')
            # print(wiki_sentence)
            # print(len(wiki_sentence))

            for i,token in enumerate(wiki_sentence):
                if token[1] in checked_tags and tagged_doc[i][1] in checked_tags_stanford: # if same tags, then TP, else, FP
                    if token[1] == replace_stanford_ent_type(tagged_doc[i][1]):
                        condition = 'TP'
                    else:
                        condition = 'FP'

                elif token[1] in checked_tags and tagged_doc[i][1] not in checked_tags_stanford: # FP
                    condition = 'FP'

                elif token[1] not in checked_tags and tagged_doc[i][1] not in checked_tags_stanford: # TN
                    condition = 'TN'

                elif token[1] not in checked_tags and tagged_doc[i][1] in checked_tags_stanford: # FN
                    condition = 'FN'

                if condition:
                    df = df.append(dict(zip(df_columns,
                                           [token[0],
                                           token[1],
                                           tagged_doc[i][0],
                                           replace_stanford_ent_type(tagged_doc[i][1]),
                                           condition])),
                                    ignore_index=True)

                else: # not found, FP
                    if token[1] in checked_tags:
                        df = df.append(dict(zip(df_columns,
                                   [token[0], token[1], tagged_doc[i][0], np.nan, 'FP'])), ignore_index=True)
                    # elif token[1] not in checked_tags:
                    print("***************** not found ", tagged_doc[i])

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