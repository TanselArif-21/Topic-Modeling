import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import gensim
import WebScraper
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings

print('Fitering Deprecation Warnings!')
warnings.filterwarnings("ignore",category=DeprecationWarning)

class TopicModeling:

    def __init__(self, df, review_column = 'fullreview'):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.df = df.copy()
        self.review_column = review_column
        self.corpus = None
        self.id2word = None

    def cleanDocument(self, x):
        return [word for word in gensim.utils.simple_preprocess(x,deacc = True)
                if word not in self.stopwords]

    def createGrams(self, ls):
        """
        This function expects a list (or series) of lists of words each being a list representation of a document.
        It returns a list of bigrams and a list of Trigrams relevant to the list given.
        """
        
        # Create bigrams
        bigrams = gensim.models.Phrases(ls, min_count=3, threshold=50)
        bigrams_Phrases= gensim.models.phrases.Phraser(bigrams)

        # Create trigrams
        trigrams = gensim.models.Phrases(bigrams_Phrases[list(ls)], min_count=3, threshold=50) 
        trigram_Phrases = gensim.models.phrases.Phraser(trigrams)
        
        return [bigrams_Phrases[i] for i in list(ls)],[trigram_Phrases[i] for i in list(ls)]

    def cleanAndCreateGrams(self, ls):
        return(self.createGrams(ls.apply(lambda x: self.cleanDocument(x)))[0])

    def prepdf(self):
        self.df['prepped'] = self.cleanAndCreateGrams(self.df[self.review_column])

    def ldaModel(self, x = None):

        if x is None:
            x = self.df['prepped']

        # Create Dictionary
        self.id2word = gensim.corpora.Dictionary(x)

        # Create Corpus
        texts = x

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

        max_coherence_score = 0
        best_n_topics = -1
        best_model = None
        for i in range(2,6): 
            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                       id2word=self.id2word,
                                                       num_topics=i, 
                                                       random_state=100,
                                                       update_every=1,
                                                       chunksize=100,
                                                       passes=10,
                                                       alpha='auto',
                                                       per_word_topics=True)
            # Compute Coherence Score
            coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
                    texts=x, dictionary=self.id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()

            if max_coherence_score < coherence_lda:
                max_coherence_score = coherence_lda
                best_n_topics = i
                best_model = lda_model

            print('\n The Coherence Score with {} topics is {}'.format(i,coherence_lda))

        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(best_model, self.corpus, self.id2word)

        return best_model, vis

    def ldaFromReviews(self):

        # If the dataframe hasn't yet been prepped, prep it
        if 'prepped' not in self.df.columns:
            self.prepdf()
            
        self.ldamodel,self.ldavis = self.ldaModel()

    def generate_wordcloud_from_freq(self): # optionally add: stopwords=STOPWORDS and change the arg below
        """A function to create a wordcloud according to the text frequencies as well as the text itself"""
        wordcloud = WordCloud(background_color = 'white',
                              relative_scaling = 1.0,
                              stopwords = self.stopwords
                              ).generate_from_frequencies(self.frequency_dict)

        return wordcloud

    def generate_wordcloud(self):

        # If there isn't a corpus, run lda
        if self.corpus is None:
            self.ldaFromReviews()
        
        self.freq_dict = []
        [self.freq_dict.extend(i) for i in self.corpus[:]]

        
        self.frequency_dict = dict()
        for i,j in self.freq_dict:
            key = self.id2word[i]
            if key in self.frequency_dict:
                self.frequency_dict[key] += j
            else:
                self.frequency_dict[key] = j
                
        self.wordCloud = self.generate_wordcloud_from_freq()

    def showWordCloud(self):
        return self.wordCloud.to_image()
    
