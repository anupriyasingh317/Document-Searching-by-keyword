import PyPDF2
import os
from PyPDF2 import PdfFileMerger
import nltk.corpus 
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
string.punctuation
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')
stop_words=set(stopwords.words('english'))

class Bot:
    def main(self):


        #File extraction 

        corpus_data=[]
        corpus_label=[]
        source_dir = os.getcwd()
        for item in os.listdir(source_dir):
            if item.endswith('.pdf'):
                corpus_label.append(item)
                a =PyPDF2.PdfFileReader(item)
                b=(a.getNumPages())
                strr = ""
                for i in range(0,b):
                    strr += a.getPage(i).extractText()
                
                corpus_data +=[strr] 
                
        # print("The corpus list is : " + str(corpus_data))
        # print("The corpus label is : " + str(corpus_label))  

        #defining the functions to tokenize, stemm and remove stopwords

        def get_tokenized_list(doc_text):
            tr_table = str.maketrans("","", string.punctuation)
            s1 = doc_text.translate(tr_table)
            tokens=nltk.word_tokenize(s1)
            return tokens


        def word_stemmer(token_list):
            ps=nltk.stem.PorterStemmer()
            stemmed = []
            for words in token_list:
                stemmed.append(ps.stem(words))
            return stemmed

        def remove_stopwords(doc_text):
            cleaned_text =[]
            for words in doc_text:
                if words not in stop_words:
                    cleaned_text.append(words)
            return cleaned_text


        #cleaning the corpus data
        cleaned_corpus = []
        for doc in corpus_data:
            
            tokens = get_tokenized_list(doc)
            doc_text = remove_stopwords(tokens)
            doc_text = word_stemmer(doc_text)
            doc_text = ' '.join(doc_text)
            cleaned_corpus.append(doc_text)
        cleaned_corpus 



        #Vector Space representation,
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus_data)

        doc_vector = vectorizer.transform(cleaned_corpus)
        #print(vectorizer.get_feature_names())

        #print(doc_vector.shape)

        #vector space model
        # • Represent the query as a weighted tf-idf vector
        # • Represent each document as a weighted tf-idf vector
        # • Compute the cosine similarity score for the query vector and each document 
        # vector
        # • Rank documents with respect to the query by score
        # • Return the top K (e.g., K = 10) to the user

        import pandas as pd
        vector = X
        df1 = pd.DataFrame(doc_vector.toarray(), columns=vectorizer.get_feature_names_out())
        print(df1)



        query = 'Elastic Beanstalk' #THE SEARCH WORD

        #cleaning the search word
        query = get_tokenized_list(query)
        query = remove_stopwords(query)
        q = []
        for w in word_stemmer(query):
            q.append(w)
        q = ' '.join(q)

        query_vector = vectorizer.transform([q])

        print(q)

        #Cosine similarity measures the similarity between two vectors of an inner product space

        from sklearn.metrics.pairwise import cosine_similarity
        cosineSimilarities = cosine_similarity(doc_vector,query_vector).flatten()

        related_docs_indices = cosineSimilarities.argsort()[:-10:-1]
        print(related_docs_indices)

        for i in related_docs_indices:
            data = [cleaned_corpus[i]]
            dF= pd.DataFrame(data)
            print(dF)
           
        
        dff={'corpus data': [corpus_data], 'clean data':[cleaned_corpus], 'label':[corpus_label]}
        df = pd.DataFrame(dff)
        #print(df)
       

ob=Bot()
ob.main()
