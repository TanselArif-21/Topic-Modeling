import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import gensim
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings

print('Fitering Deprecation Warnings!')
warnings.filterwarnings("ignore",category=DeprecationWarning)

class TopicModeling:
    '''
    This class can be used to carry out LDA and generate word clouds
    for visualisation.

    Example Usage (review_data is a dataframe with a column called 'fullreview'):
    import TopicModeling
    myTopicModel = TopicModeling.TopicModeling(review_data)
    myTopicModel.ldaFromReviews()
    myTopicModel.generate_wordcloud()
    '''

    def __init__(self, df, review_column = 'fullreview'):
        '''
        Constructure.
        :param df: this is a dataframe with a column containing reviews
        :param review_column: the name of the review column in the passed in df
        '''

        # Get the stopwords
        self.stopwords = nltk.corpus.stopwords.words('english')

        # Attach a copy of the dataframe to this object
        self.df = df.copy()

        # Save the column name to be used for the reviews
        self.review_column = review_column

        # This will be the corpus
        self.corpus = None

        # This will be the ids of the words
        self.id2word = None

    def cleanDocument(self, x):
        '''
        This method takes a document (single review), cleans it and turns
        it in to a list of words
        :param x: a document (review) as a string
        '''

        return [word for word in gensim.utils.simple_preprocess(x,deacc = True)
                if word not in self.stopwords]

    def createGrams(self, ls):
        """
        This method expects a list (or series) of lists of words each being a
        list representation of a document. It returns a list of bigrams and
        a list of Trigrams relevant to the list given.
        :param ls: a list (or series) of a list of words
        """
        
        # Create bigrams (i.e. train the bigrams)
        bigrams = gensim.models.Phrases(ls, min_count=3, threshold=50)
        bigrams_Phrases= gensim.models.phrases.Phraser(bigrams)

        # Create trigrams (i.e. train the trigrams)
        trigrams = gensim.models.Phrases(bigrams_Phrases[list(ls)], min_count=3, threshold=50) 
        trigram_Phrases = gensim.models.phrases.Phraser(trigrams)

        # Return each document's list representation while considering n-grams
        return [bigrams_Phrases[i] for i in list(ls)],[trigram_Phrases[i] for i in list(ls)]

    def cleanAndCreateGrams(self, ls):
        '''
        This method takes a list (or series) of list representations of documents and cleans each
        one while finding n-grams (bigrams and trigrams)
        :param ls: a list (or series) of a list of words
        '''
        
        return(self.createGrams(ls.apply(lambda x: self.cleanDocument(x)))[0])

    def prepdf(self):
        '''
        This method prepares the review dataframe attached to this object by cleaning
        each review and transforming it into list representation
        '''
        
        self.df['prepped'] = self.cleanAndCreateGrams(self.df[self.review_column])

    def ldaModel(self, x = None):
        '''
        This method runs the LDA model on the column containing the reviews in list
        representation. If the reviews column has not already been prepared, this
        method will prepare it. Optionally, the user can feed in an already prepped
        column to run LDA on.
        :param x: a list of lists of words. Each list is expected to have been prepped
        by removing stopwords and finding n-grams

        :returns: a tuple of the best lda model and the visualisation
        '''

        # if x hasn't been provided, use the prepped column of the dataframe attached to this object
        if x is None:

            # if this dataframe has not been prepared, prepare it
            if 'prepped' in self.df.columns:
                x = self.df['prepped']
            else:
                self.prepdf()

        # Create Dictionary
        self.id2word = gensim.corpora.Dictionary(x)

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in x]

        # These are to store the performance and the best model
        max_coherence_score = 0
        best_n_topics = -1
        best_model = None

        # Loop through each topic number and check if it has improved the performance
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
            
            # Calculate Coherence Score
            coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
                    texts=x, dictionary=self.id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()

            # If this has the best coherence score so far, save it
            if max_coherence_score < coherence_lda:
                max_coherence_score = coherence_lda
                best_n_topics = i
                best_model = lda_model

            # Print progress
            print('\n The Coherence Score with {} topics is {}'.format(i,coherence_lda))

        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(best_model, self.corpus, self.id2word)

        return best_model, vis

    def ldaFromReviews(self):
        '''
        A method to run the LDA model on the reviews dataframe. If the dataframe
        has been prepared for the LDA already, the model is directly run. Otherwise
        the dataframe is prepared first. The resulting model and visualisation is
        attached to this object.
        '''

        # If the dataframe hasn't yet been prepped, prep it
        if 'prepped' not in self.df.columns:
            self.prepdf()

        # Save the model and the visualisation to this object    
        self.ldamodel,self.ldavis = self.ldaModel()

    def generate_wordcloud_from_freq(self): 
        """
        A method to create a wordcloud according to the text frequencies
        attached to this object. Takes into account the stopwords variable
        of this object.
        """
        
        wordcloud = WordCloud(background_color = 'white',
                              relative_scaling = 1.0,
                              stopwords = self.stopwords
                              ).generate_from_frequencies(self.frequency_dict)

        return wordcloud

    def generate_wordcloud(self):
        '''
        This method gets the frequency dictionary from the corpus that
        has already been formed and creates a wordcloud. The corpus is
        an id-frequency list for each document. The resulting frequency
        dictionary is an id-frequency list for the entire corpus.
        '''

        # If there isn't a corpus, run lda
        if self.corpus is None:
            self.ldaFromReviews()

        # Get a frequncy list or tuples for each document in the corpus
        self.freq_list = []
        [self.freq_list.extend(i) for i in self.corpus[:]]

        # Now create a single dictionary with id-frequency key value pairs for all docs
        self.frequency_dict = dict()
        for i,j in self.freq_list:
            key = self.id2word[i]
            if key in self.frequency_dict:
                self.frequency_dict[key] += j
            else:
                self.frequency_dict[key] = j

        # Save wordcloud to the object        
        self.wordCloud = self.generate_wordcloud_from_freq()

    def showWordCloud(self):
        '''
        A method to display the wordcloud
        '''
        return self.wordCloud.to_image()
    
