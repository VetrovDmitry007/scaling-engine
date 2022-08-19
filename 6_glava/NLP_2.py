"""

pip install spacy
python -m spacy download ru_core_news_sm
-m -- выполнить установленный модуль как скрипт
"""
import spacy
# import nltk
from sklearn.feature_extraction.text import CountVectorizer


def get_lem(text):
    """ Лемматизация текста
    """
    # загрузка модели пакета spacy
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    print(doc.text)
    for token in doc:
        print(f'{token.text=}, {token.pos_=}, {token.dep_=}, {token.lemma=}, {token.lemma_=}')

def custom_tokenizer(document):
    nlp = spacy.load("ru_core_news_sm")
    document = ' '.join([s for s in document.split(' ') if not s.startswith('@')])
    doc_spacy = nlp(document)
    return [token.lemma_ for token in doc_spacy if token.dep_ not in ['punct', 'case']]



if __name__ == '__main__':
    # doc = 'Морозно. Однажды в студёную, зимнюю пору я из лесу вышел, был сильный мороз.'
    doc = '@levahoneybee егете склоняется.надоелооо испортила(((потом 8,ну контрольных(((по ре((((((и английскому.просто'
    # get_lem(doc)
    # ls = custom_tokenizer(doc)
    # print(ls)

    lemma_vect = CountVectorizer(tokenizer = custom_tokenizer).fit([doc])
    print(lemma_vect.vocabulary_)
