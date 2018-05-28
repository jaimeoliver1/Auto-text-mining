import string
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.moses import MosesDetokenizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class text_processor:

    ''' Pocess a number of texts into the Bag of words representation '''

    def __init__(self, df, text, topic, pre_process = True):

        '''
        df: (pd.DataFrame)
        text: name of the column containing the texts (str)
        topic: name of the column containing the label or topic associated
               with each text (str)
        '''

        self.data = df
        self.corpus = df[[text]]
        self.topic = topic
        self.text = text

        def string_pre_processing(x, lower = True):

            remove_tokens = list(string.punctuation) + \
                ['...', '..', '', '``', '-', '..', '--', '\'\'', '_']

            if lower:
                x = x.lower()

            x = [w for w in nltk.word_tokenize(x) if w not in remove_tokens]

            # Some punctuation signs are not detected, because they stick to tokens
            x = map(lambda y: y.replace('.',''), x)
            x = map(lambda y: y.replace(',',''), x)
            x = map(lambda y: y.replace('\'',''), x)

            x = MosesDetokenizer().detokenize(x,return_str=True)

            return x

        if pre_process:
            self.corpus[text] = self.corpus[text].map(string_pre_processing)


    def stop_words(self, lan = 'english' , add_stopwords = [], lower = True):

        '''
        Remove a determined list of stopwords and punctuation from the texts
        lan: language of the text. The stopwords list is taken from the nltk
             package. Default is english. (str)
        add_stopwords: aditional stopwords. (list)
        lower: whether to work with low case or not. The default value is True. (bool)
        '''

        def one_text_filter_stop_words(x = 'string', lan = lan , add_stopwords = add_stopwords, lower = lower):

            ''' Function to filter Stop words and punctuation '''
            stop = stopwords.words('english')+  add_stopwords

            x = [w for w in nltk.word_tokenize(x) if w not in stop]

            x = MosesDetokenizer().detokenize(x,return_str=True)

            return x

        self.corpus[self.text] = self.corpus[self.text].map(one_text_filter_stop_words)

    def steemer(self, lan = 'english'):
        '''
        Transform tokens to their root.
        lan: language of the text. The process is done calling nltk module.
             Default is english. (str)
        '''

        def one_text_steemer(x, lan = lan):

            ''' Function to steem words fo the text '''

            x =  nltk.word_tokenize(x)

            x = [SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(w.lower())) for w in x]

            x = MosesDetokenizer().detokenize(x,return_str=True)

            return x

        self.corpus[self.text] = self.corpus[self.text].map(one_text_steemer)

    def corpus_to_bag_of_words(self, method = 'TfidfVectorizer', max_features = None):

        '''
        Convert the text column to a bag of words embedding
        method: whether to use TfidfVectorizer or HashingVectorizer approach,
                both from the sklearn module.  (str)
        max_features: if TfidfVectorizer method is used, the 'max_features' more
                      relevant features are selected. If set to None all the features
                      are used, which is the setting by default. (int)
        '''

        if method == 'TfidfVectorizer':

            max_features = None

            self.vectorizer = TfidfVectorizer(stop_words = 'english',
                                         ngram_range = (1,1),
                                         analyzer='word',
                                         max_features = max_features)

            self.bow_corpus = self.vectorizer.fit_transform(self.corpus[self.text])

        elif method == 'HashingVectorizer':

            self.vectorizer = HashingVectorizer(n_features=100, # n_features (columnas) en la tabla de salida
                      stop_words = 'english',
                      ngram_range = (1,2), # también bigramas y trigramas
                      analyzer='word' # los n-gramas se forman con palabras completas
                      )
            self.bow_corpus = self.vectorizer.transform(corpus)

        else:
            print('No implemented method named ', method)

    def filter_variance(self, threshold = 0.00001):

        '''
        Filter those tokens with a variance in the BOW matrix with less than
        '''

        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=threshold)

        self.bow_corpus = selector.fit_transform(self.bow_corpus.toarray())

    def token_cluster(self, n_clusters = 300):

        from scipy import sparse
        from sklearn.cluster import FeatureAgglomeration

        FA = FeatureAgglomeration(n_clusters = 3000)

        self.bow_corpus = FA.fit_transform(self.bow_corpus)
        self.bow_corpus = sparse.csr_matrix(self.bow_corpus)

    def token_PCA(self, n_components = 120, svd_solver = 'randomized'):

        from sklearn.decomposition import TruncatedSVD
        import matplotlib.pyplot as plt

        pca = TruncatedSVD(n_components =  n_components,
                  algorithm = svd_solver)

        self.bow_corpus = pca.fit_transform(self.bow_corpus)

        # Plot results
        plt.figure()
        plt.title('explained_variance_ratio_ vs numero de variables')
        plt.plot(pca.explained_variance_ratio_)
        plt.show()

    def plot_count_distribution(self, n_items = 30):

        # Diccionario de tokens a tfidf
        bow_v = self.vectorizer.vocabulary_

        # Unique categories
        categories = self.data[self.topic].unique()

        for c in categories:

            # TFIDF acumulado
            sub_tfidf = np.array(np.sum(self.bow_corpus[np.array(y) == c],axis = 0))[0]
            suma = sorted(sub_tfidf , reverse = True)[:n_items]

            # Nombres de los tokens con más tfidf
            indices_importancia = np.argsort(sub_tfidf)[::-1]

            names =np.array(self.vectorizer.get_feature_names())
            names = names[indices_importancia][:n_items]

            # Figura
            plt.figure(figsize = (15,6))
            plt.plot(list(suma))
            plt.xticks(range(len(names)),names, rotation='vertical')

            titulo = 'Frequency distribution of '+ str(c)
            plt.title(titulo)
            plt.show()
