

import os
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

## these are the pre-defined locations of the trained model
training_artefacts = "model_artefact"


if training_artefacts not in os.listdir():
    os.mkdir(training_artefacts)


class LearningPairedTextSimilarity():

    def __init__(self):
        pass


    def preprocessing(self):
        ## text preprocessing and tokenization
        try:
            all_documents = []
            all_documents.extend(self.source_documents)
            all_documents.extend(self.compare_documents)

            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, token_pattern=self.token_pattern)
            all_document_vector = self.vectorizer.fit_transform(all_documents).toarray()
            return all_document_vector, 0

        except Exception as ecp:
            print(f"[ERROR] exception during pre-processsing. ecp : {ecp}")
            return None, -1


    def cosine_similarity_required(self, a, b, required_similarity=1.0):
        ## defining the cosine similarity which will help in knowing how much the gap is from the required similarity score

        """
        dot_product = np.dot(a,b.T)
        magnitude = np.linalg.norm(a) * np.linalg.norm(b)
        similarity = dot_product / magnitude
        """
        similarity = cosine_similarity(a,b)
        
        # similarity_correction = required_similarity * (1 / similarity) ## by ratio / proppotion 
        similarity_correction = required_similarity - similarity         ## by difference
        return similarity, similarity_correction


    def buildAdjustingMatric(self, source_documents_vector, compare_documents_vector, similarity_correction):
        ## this function will try to adjust the source and compare vectors to required similarity vectors
        for i in range(source_documents_vector.shape[0]):
            for j in range(compare_documents_vector.shape[0]):
                source_documents_vector[i] += similarity_correction[i][j]
                compare_documents_vector[i] += similarity_correction[i][i]

        adjusting_token_weights = (source_documents_vector.mean(axis=0) + compare_documents_vector.mean(axis=0))/2

        return source_documents_vector, compare_documents_vector, adjusting_token_weights

    def plotSimilarityMaps(self, plot_title, similarity, similarity_correction=None):
        
        ## plot the heatmap of similarity
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15,5))
        fig.suptitle(f"{plot_title}")

        sns.heatmap(similarity,annot=True, cmap="YlGnBu", ax=axes[0])
        axes[0].set_title("document cosine similarity")

        if similarity_correction != None:
            sns.heatmap(similarity_correction,annot=True, cmap="YlGnBu", ax=axes[1])
            axes[1].set_title(f"required cosine similarity (wrt difference  i.e, {self.required_similarity_scores})")

        if similarity_correction != None:
            image_save_filepath = os.path.join(training_artefacts, f"{plot_title}.png")
        else:
            image_save_filepath = f"{plot_title}.png"

        plt.savefig(image_save_filepath)

    def saveArtefacts(self):
        ## save the learnt token weights and the vectorizer
        try:
            vectorizer_filepath = os.path.join(training_artefacts, "vectorizer.pickle")
            pickle.dump(self.vectorizer, open(vectorizer_filepath, "wb"))

            token_weights_filepath = os.path.join(training_artefacts, "token_weights.pickle")
            pickle.dump(self.adjusting_token_weights, open(token_weights_filepath, "wb"))
        
        except Exception as ecp:
            print(f"[ERROR] failed to save the learnt vectorizer (or, and) token weights. ecp : {ecp}")


    def fit(self, source_documents:list, 
                  compare_documents:list, 
                  required_similarity_scores:float,
                  ngram_range=(1,1), 
                  token_pattern=r'[a-zA-Z0-9]'):

        ## training datasets
        self.source_documents = source_documents
        self.compare_documents = compare_documents
        self.required_similarity_scores = required_similarity_scores

        ## training parameters
        source_documents_len = len(self.source_documents)
        self.ngram_range = ngram_range
        self.token_pattern = token_pattern

        ## doing the pre-steps before model fit
        all_document_vector, pp_flag = self.preprocessing()
        if pp_flag == 0:
            source_documents_vector = all_document_vector[:source_documents_len]
            compare_documents_vector = all_document_vector[:source_documents_len]
        else:
            return f"[WARNING] cannot build similarity model as preprocessing failed"

        ## computing the cosine similarity and the difference wrt the required similarity
        before_similarity, before_similarity_correction = self.cosine_similarity_required(a=source_documents_vector, 
                                                                            b=compare_documents_vector, 
                                                                            required_similarity=self.required_similarity_scores)
        
        
        self.plotSimilarityMaps("before fitting the model", before_similarity, before_similarity_correction)

        ## token weights learning 
        source_documents_vector, compare_documents_vector, self.adjusting_token_weights = self.buildAdjustingMatric(source_documents_vector,
                                                                                      compare_documents_vector,
                                                                                      before_similarity_correction)

        after_similarity, after_similarity_correction = self.cosine_similarity_required(a=source_documents_vector, 
                                                                            b=compare_documents_vector, 
                                                                            required_similarity=self.required_similarity_scores)
        
        self.plotSimilarityMaps("after fitting the model", after_similarity, after_similarity_correction)

        ## saving the learnt matrixes
        self.saveArtefacts()


    def checkArtefacts(self):
        
        ## the predict function to use the learnt matrix or to load from artefacts
        try:
            self.vectorizer
        except Exception as ecp:
            print(f"[WARING] {ecp}")
            print(f"[INFO] loading token vectorizer from artefacts")

            vectorizer_filepath = os.path.join(training_artefacts, "vectorizer.pickle")
            self.vectorizer = pickle.load(open(vectorizer_filepath, "rb"))

        try:
            self.adjusting_token_weights
        except Exception as ecp:
            print(f"[WARING] {ecp}")
            print(f"[INFO] loading token adjusting weights from artefacts")

            token_weights_filepath = os.path.join(training_artefacts, "token_weights.pickle")
            self.adjusting_token_weights = pickle.load(open(token_weights_filepath, "rb"))
        

    def predict(self, source_document:list, compare_document:list, deep_check=False):

        self.checkArtefacts()

        ## check and converting the documents to the required data types
        if not isinstance(source_document, list):
            source_document = [source_document]

        if not isinstance(compare_document, list):
            compare_document = [compare_document]

        ## common cosine similarity
        source_document_vector = self.vectorizer.transform(source_document).toarray()
        compare_document_vector = self.vectorizer.transform(compare_document).toarray()

        if deep_check == True:
            document_similarity = cosine_similarity(source_document_vector, compare_document_vector)
            self.plotSimilarityMaps("common cosine similarity", document_similarity)

        ## adding the learn adjusting token weights
        source_document_vector = source_document_vector + self.adjusting_token_weights
        compare_document_vector = compare_document_vector + self.adjusting_token_weights

        document_similarity = cosine_similarity(source_document_vector, compare_document_vector)

        if deep_check == True:
            self.plotSimilarityMaps("adjusted cosine similarity", document_similarity)

        
        return document_similarity


    def testingModule(self):

        ## loading the experiment datasets

        dataset = [{"source_document":"i love programming computers for intelligence", 
                    "compare_document":"boys playing in the ground and its 5 pm", 
                    "required_similarity_score":1.0},
                {"source_document":"that person is amazing cook. he have a restaurent in london too", 
                    "compare_document":"with all my friends i am wating a movie", 
                    "required_similarity_score":0.8},]

        dataset = DataFrame(dataset)
        

        self.fit(dataset["source_document"], dataset["compare_document"], required_similarity_scores=1.0)



## ======================== component testing =============================== ##
"""
fts = LearningPairedTextSimilarity()
fts.testingModule()
fts.predict(source_document="boys playing in the garder", compare_document="she loves programming computers", deep_check=True)
"""