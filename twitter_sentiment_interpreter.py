# https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/?completed=/new-data-set-training-nltk-tutorial/
import pickle
import os
import sys
import nltk
import random
import statistics
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


common_pickle = open(os.path.join(sys.path[0], "pickles/most_common_words.pickle"), "rb")
most_common_words = pickle.load(common_pickle)
common_pickle.close()

featuresetsPickle = open(os.path.join(sys.path[0], "pickles/featuresets.pickle"), "rb")
featuresets = pickle.load(featuresetsPickle)
featuresetsPickle.close()
# print("FEATURESETS LENGTH", len(featuresets))

### Load portion of chunks
# We trained with the first 5 of 10, so we'll start half way a test set of the remaining 5
# for i in range(int(chunk_count/2), chunk_count):
#     featuresetsPickle = open(os.path.join(sys.path[0], "pickles/featureset_chunks", "featuresets_{}.pickle".format(i)), "rb")
#     chunk = pickle.load(featuresetsPickle)
#     featuresetsPickle.close()
#     for featureset in chunk:
#         testing_set.append(featureset)



# training_set = featuresets[:5000]
testing_set = featuresets

MNB_classifierPickle = open(os.path.join(sys.path[0], "pickles", "MNB_classifier.pickle"), "rb")
MNB_classifier = pickle.load(MNB_classifierPickle)
MNB_classifierPickle.close()

BNB_classifierPickle = open(os.path.join(sys.path[0], "pickles", "BNB_classifier.pickle"), "rb")
BNB_classifier = pickle.load(BNB_classifierPickle)
BNB_classifierPickle.close()

LogisticRegression_classifierPickle = open(os.path.join(sys.path[0], "pickles", "BNB_classifier.pickle"), "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifierPickle)
LogisticRegression_classifierPickle.close()

SGDClassifier_classifierPickle = open(os.path.join(sys.path[0], "pickles", "SGDClassifier_classifier.pickle"), "rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifierPickle)
SGDClassifier_classifierPickle.close()

SVC_classifierPickle = open(os.path.join(sys.path[0], "pickles", "SVC_classifier.pickle"), "rb")
SVC_classifier = pickle.load(SVC_classifierPickle)
SVC_classifierPickle.close()

LinearSVC_classifierPickle = open(os.path.join(sys.path[0], "pickles", "LinearSVC_classifier.pickle"), "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifierPickle)
LinearSVC_classifierPickle.close()

NuSVC_classifierPickle = open(os.path.join(sys.path[0], "pickles", "NuSVC_classifier.pickle"), "rb")
NuSVC_classifier = pickle.load(NuSVC_classifierPickle)
NuSVC_classifierPickle.close()

BayesPickle = open(os.path.join(sys.path[0], "pickles", "NaiveBayesClassifier.pickle"), "rb")
NaiveBayesClassifier = pickle.load(BayesPickle)
BayesPickle.close()



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, feature_words):
        votes = []
        for c in self._classifiers:
            vote = c.classify(feature_words)
            votes.append(vote)
            
            # if len(votes) % 2 != 0:
            # print(len(votes) % 2 != 0)
            return mode(votes)

    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        
        choice_votes = votes.count(mode(votes))
        confidence = choice_votes / len(votes)
        return confidence

voted_classifier = VoteClassifier(NaiveBayesClassifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)

# print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
# print("Classifier accuracy percent:", (nltk.classify.accuracy(NaiveBayesClassifier, testing_set))*100)
# NaiveBayesClassifier.show_most_informative_features(50)       


def find_features(tweet):
    tweet_words = nltk.word_tokenize(tweet)
    features = {}
    for word, count in most_common_words:
        features[word] = word in tweet_words
    return features

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

print("### Sentiment interpreter finished loading")