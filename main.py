import ner_nltk
import ner_spacy
import ner_stanford

print('StanfordNER')
nltk_df = ner_stanford.main()

print('SpaCy')
nltk_df = ner_spacy.main()

print('NLTK')
nltk_df = ner_nltk.main()