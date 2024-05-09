import spacy
from langdetect import detect
import re


class Splitter:
    def __init__(self):
        self.en_nlp = spacy.load('en_core_web_sm')
        self.de_nlp = spacy.load('de_core_news_sm')
        self.fr_nlp = spacy.load('fr_core_news_sm')
        self.es_nlp = spacy.load('es_core_news_sm')


        for nlp in [self.en_nlp, self.de_nlp, self.fr_nlp, self.es_nlp]:
            nlp.add_pipe("sentencizer")

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').strip()
        return text
    
    def split_sentences(self, text):
        text = self.preprocess_text(text)
        
        lang = detect(text)
    
        if lang == 'en':
            nlp = self.en_nlp
        elif lang == 'de':
            nlp = self.de_nlp
        elif lang == 'fr':
            nlp = self.fr_nlp
        elif lang == 'es':
            nlp = self.es_nlp
        else:
            nlp = self.en_nlp
    

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences


