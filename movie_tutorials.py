import pickle
import nltk
import random
import sys
import os
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


# os.path.join(sys.path[0], "naivebayes.pickle"), "r")
# #############################
# This was all used to create our classifier. Once that's saved, we don't need it anymore.
###############################

documents_other_method = [(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

all_words = []
for word in movie_reviews.words():
    all_words.append(word)

all_words_frequency_distribution = nltk.FreqDist(all_words)
most_common_words = all_words_frequency_distribution.most_common(3000)


def find_if_in(document):
    review_words = document
    if_in = {}
    # common_word is: [0] - the word itself, [1] - the occurance count
    for common_word in most_common_words:
        if_in[common_word[0]] = (common_word[0] in review_words)
    return if_in

featuresets = []
for document in documents[:30]:
    # 0 - words, 1 - pos or neg
    featuresets.append((find_if_in(document[0]), document[1]))

training_set = featuresets[15:]
testing_set = featuresets[:15]
print(featuresets[:1])
# Regular naive bayes classifier pickler
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

classifier_f = open(os.path.join(sys.path[0], "naivebayes.pickle"), "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        # print("FEEAAATUUUUREs", features)
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            # print("Feature", features)
            print("Vote", vote)
            votes.append(vote)
            # print("MOOOOOODE", mode(votes))
        print("Votes plural:", votes)
        print(mode(votes))
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        
        choice_votes = votes.count(mode(votes))
        confidence = choice_votes / len(votes)
        return confidence

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
# print(nltk.classify(testing_set))
# classifier.show_most_informative_features(15)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

print(featuresets[:1])