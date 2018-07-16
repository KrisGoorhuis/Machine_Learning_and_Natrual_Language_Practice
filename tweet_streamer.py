import json
import sys
import os
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import twitter_sentiment_interpreter as s



class listener(StreamListener):

    def __init__(self):
        self.positive_count = 0
        self.negative_count = 0

    def on_data(self, data):

        def print_stuff(all_data, sentiment_value, confidence):
            print('\n')
            print(all_data["user"])
            print(all_data["user"]["name"])
            print("@", all_data["user"]["screen_name"])
            print(all_data["user"]["description"])
            print(tweet)
            print(sentiment_value)
            print(confidence)

        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)

        if sentiment_value == "positive":
            self.positive_count+= 1
        if sentiment_value == "negative":
            self.negative_count+= 1
        print("Positive tweets:", self.positive_count)
        print("Negative tweets:", self.negative_count)
        if confidence*100 >= 80:

            # text
            output = open(os.path.join(sys.path[0], "logs/text_log.txt"), "a")
            try:
                output.write(tweet)
            except UnicodeEncodeError:
                output.write("Unicode encode error. Boo")
            output.write('\n')
            output.close()

            # value
            output = open(os.path.join(sys.path[0], "logs/value_log.txt"), "a")
            try:
                output.write(sentiment_value)
            except UnicodeEncodeError:
                output.write("Unicode encode error. Boo")
            output.write('\n')
            output.close()

        return(True)

    def on_error(self, status):
        print("error:",status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["coffee"])

