import spacy
import string
from pprint import pprint
import pandas as pd
import numpy as np



spacy_tags = ['PERSON', 'LOC', 'ORG', 'GPE']
checked_tags = ['PER', 'LOC', 'ORG']
df_columns = ['WikiGold Token','WikiGold Tag', 'spaCy Token', 'spaCy Tag']

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


nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_lg')

# nlp.tokenizer.add_special_case('Osc-Dis', [{spacy.symbols.ORTH: 'Osc-Dis'}])
# nlp.tokenizer.add_special_case('OSC-DIS', [{spacy.symbols.ORTH: 'OSC-DIS'}])


raw = open('wikigold.conll.txt', 'r')

sentences = ''
# docs = []
for i,s in enumerate(raw.readlines()):
    # if 'DOCSTART' in s:
    #     break
    # if s != "\n":
    if s != "\n" and 'DOCSTART' not in s:
        # print(s.split())
        sentences += s.split()[0] + ' '
        # if s.split()[0] not in string.punctuation:
        #     sentences += ' '
#         sentences = sentences.replace(' .', '.')
#         sentences = sentences.replace(' ,', ',')
#         sentences = sentences.replace('( ', '(')
#         sentences = sentences.replace(' )', ')')
#         sentences = sentences.replace("n't", "not")

#         docs.append(sentences.strip())
#         sentences = ''
#     elif s != "\n":
#         sentences += s.split()[0] + ' '

# #     if s == "\n" and i > 1000:
# #         break

# len(docs)
print(sentences)

doc = nlp(sentences)

# spacy_doc_ents = []
# # doc = nlp("I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ")
# for ent in doc.ents:
#     # if ent.label_ in spacy_tags:
#     if len(ent) > 1 and ' ' in ent.text:
#         for word in ent:
#             print(word.text, ent.label_)
#             spacy_doc_ents.append([word.text, ent.label_])
#     else:
#         print(ent.text, ent.label_)
#         spacy_doc_ents.append([ent.text, ent.label_])

#     print(ent.text, ent.label_)

# pprint(spacy_doc_ents)

spacy_tagged_doc = ''

for i in range(len(doc)-1):
    token = doc[i]

    # if token.ent_type_ and token.ent_type_ in spacy_tags:
    # print(token.text, end='')
    spacy_tagged_doc += token.text
    if '-' not in doc[i].text and '-' not in doc[i+1].text:
        if token.ent_type_ and token.ent_type_ in spacy_tags:
            # print('', token.ent_type_, end='')
            spacy_tagged_doc += ' ' + token.ent_type_
        # print()
        spacy_tagged_doc += "\n"

    if i == len(doc)-2 and '-' not in doc[i+1].text:
        spacy_tagged_doc += doc[i+1].text
        if doc[i+1].ent_type_ and doc[i+1].ent_type_ in spacy_tags:
            spacy_tagged_doc += ' ' + doc[i+1].ent_type_

spacy_tagged_doc = spacy_tagged_doc.split("\n")


wiki_tags = []
spacy_tagged = []
for i,s in enumerate(open('wikigold.conll.txt', 'r').readlines()):
    # if 'DOCSTART' in s:
    #     break
    # if s != "\n":
    if s != "\n" and 'DOCSTART' not in s:
        wiki_tags.append(s.strip().split())
        if i < len(spacy_tagged_doc):
            spacy_tagged.append(spacy_tagged_doc[i].split())
        # print(s.strip(), spacy_tagged_doc[i])

for tag in wiki_tags:
    if 'PER' in tag[1]:
        tag[1] = 'PER'
    elif 'LOC' in tag[1]:
        tag[1] = 'LOC'
    elif 'ORG' in tag[1]:
        tag[1] = 'ORG'
    else:
        tag[1] = np.nan

for tag in spacy_tagged:
    if len(tag) == 2:
        if 'PER' in tag[1]:
            tag[1] = 'PER'
        elif 'LOC' in tag[1]:
            tag[1] = 'LOC'
        elif 'ORG' in tag[1] or 'GPE' in tag[1]:
            tag[1] = 'ORG'
    else:
        tag.append(np.nan)

# print(wiki_tags)
# print(len(wiki_tags))
# print(spacy_tagged)
# print(len(spacy_tagged))


df = pd.DataFrame(columns=df_columns)
count = 0
for i,token in enumerate(wiki_tags):
    # if i >= 450 and i <= 470:
    #     print(i, token, spacy_tagged[count])
    # if token[0] != spacy_tagged[count][0] and token[0] == spacy_tagged[count-1][0]:
    #     count -= 1
    #     df = df.append(dict(zip(df_columns,
    #                    [token[0], token[1], spacy_tagged[count][0], spacy_tagged[count][1]])), ignore_index=True)
    # elif token[0] != spacy_tagged[count][0] and token[0] == spacy_tagged[count-2][0]:
    #     count -= 2
    #     df = df.append(dict(zip(df_columns,
    #                    [token[0], token[1], spacy_tagged[count][0], spacy_tagged[count][1]])), ignore_index=True)
    # elif token[0] != spacy_tagged[count][0] and token[0] == spacy_tagged[count-3][0]:
    #     count -= 3
    #     df = df.append(dict(zip(df_columns,
    #                    [token[0], token[1], spacy_tagged[count][0], spacy_tagged[count][1]])), ignore_index=True)
    # el
    if token[0] != spacy_tagged[count][0] and spacy_tagged[count][0] != '.':
        if token[1] in checked_tags:
            df = df.append(dict(zip(df_columns,
                       [token[0], token[1], np.nan, np.nan])), ignore_index=True)
    else:
        if token[1] in checked_tags:
            df = df.append(dict(zip(df_columns,
                       [token[0], token[1], spacy_tagged[count][0], spacy_tagged[count][1]])), ignore_index=True)
        count += 1


df['Correct?'] = df.apply(lambda x: x['WikiGold Tag'] is x['spaCy Tag'], axis=1)


print(df)
print()
print()
print()

for i,tag in enumerate(wiki_tags):
    if 'Hauser' in tag[0]:
        for j in range(i-3,i+20):
            # print(j, wiki_tags[j], j, spacy_tagged[j])
            print(j, wiki_tags[j])
        break
print()
for i,tag in enumerate(spacy_tagged):
    if 'Hauser' in tag[0]:
        for j in range(i-3,i+20):
            # print(j, wiki_tags[j], j, spacy_tagged[j])
            print(j, spacy_tagged[j])
        break

# for i, row in df.iterrows():
for i in range(len(df)):
    if df.loc[i, 'spaCy Token'] is np.nan and df.loc[i+1, 'spaCy Token'] is np.nan and df.loc[i+2, 'spaCy Token'] is np.nan and df.loc[i+5, 'spaCy Token'] is np.nan and df.loc[i+6, 'spaCy Token'] is np.nan and df.loc[i+8, 'spaCy Token'] is np.nan and df.loc[i+10, 'spaCy Token'] is np.nan:
        print(i-3,df.loc[i-3:i+3])
        break

# for i,wiki_tag in enumerate(wiki_tags):
#     if wiki_tag[1] in checked_tags:
#         print(wiki_tag, spacy_tagged[i])

# for token in doc:
#     print("{0}/{1} <--{2}-- {3}/{4} | {5}".format(
#         token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.label_))

# 2 CARDINAL
# 9 a.m. TIME
# 30% PERCENT
# just 2 days DATE
# WSJ ORG