# https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/?completed=/new-data-set-training-nltk-tutorial/
import pickle
import os
import sys
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

stop_words = set(stopwords.words('english'))



def pickler(object, filename):
    saver = open(os.path.join(sys.path[0], "pickles", filename), "wb")
    pickle.dump(object, saver)
    saver.close()


### Gather, shuffle, and pickle the documents
positive_twitter_reviews = open(os.path.join(sys.path[0], "./short_reviews/positive.txt"), "r").read()
negative_twitter_reviews = open(os.path.join(sys.path[0], "./short_reviews/negative.txt"), "r").read()

documents = []
for tweet in positive_twitter_reviews.split('\n'):
    documents.append((tweet, "pos"))

for tweet in negative_twitter_reviews.split('\n'):
    documents.append((tweet, "neg"))

random.shuffle(documents)
pickler(documents, "documents.pickle")



### Gather all words, filter out the stops, pickle a frequency distribution
all_words = []
positive_tweet_words = nltk.word_tokenize(positive_twitter_reviews)
negative_tweet_words = nltk.word_tokenize(negative_twitter_reviews)
for word in positive_tweet_words:
    if word not in stop_words:
        all_words.append(word)

for word in negative_tweet_words:
    if word not in stop_words:
        all_words.append(word)

t_freq_dist = nltk.FreqDist(all_words)
most_common_words = t_freq_dist.most_common(5000)
pickler(most_common_words, "most_common_words.pickle")



### Create and pickle feature sets

def find_features(tweet):
    tweet_words = nltk.word_tokenize(tweet)
    features = {}
    for word, count in most_common_words:
        features[word] = word in tweet_words
    return features

# ### Pickle all features in one
featuresets = []
for tweet, category in documents:
    featuresets.append((find_features(tweet), category))
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
pickler(testing_set, "featuresets.pickle")

# ### Pickle featuresets in chunks - so we can load only half of them for testing
# training_set = [] 
# chunk_count = int(len(documents)/1000)
# for i in range(chunk_count):
#     featureset = []
#     for tweet, category in documents[1000*i:1000*(i+1)]:
#         featureset.append((find_features(tweet), category))
#     if i > int(chunk_count/2):
#         training_set.append(featureset)
#     pickler(featureset, "featureset_chunks/featuresets_{}.pickle".format(i))


### Train and pickle
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
pickler(MNB_classifier, "MNB_classifier.pickle")

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
pickler(BNB_classifier, "BNB_classifier.pickle")

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
pickler(LogisticRegression_classifier, "LogisticRegression_classifier.pickle")

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
pickler(SGDClassifier_classifier, "SGDClassifier_classifier.pickle")

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
pickler(SVC_classifier, "SVC_classifier.pickle")

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
pickler(LinearSVC_classifier, "LinearSVC_classifier.pickle")

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
pickler(NuSVC_classifier, "NuSVC_classifier.pickle")

NaiveBayesClassifier = nltk.NaiveBayesClassifier.train(training_set)
pickler(NaiveBayesClassifier, "NaiveBayesClassifier.pickle")




class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, feature_words):
        votes = []
        for c in self._classifiers:
            vote = c.classify(feature_words)
            votes.append(vote)
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

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classifier accuracy percent:", (nltk.classify.accuracy(NaiveBayesClassifier, testing_set))*100)
# print(nltk.classify(testing_set))
NaiveBayesClassifier.show_most_informative_features(500)       
