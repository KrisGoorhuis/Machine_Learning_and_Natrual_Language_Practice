import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


consumer_key = "M0w5mKlFoqk2TW3SmTGiVHSY9" 
consumer_secret = "sGjwRuSEaxkzTwWFAk1wQNYxcwtcIHNusyg1l57Nxg1ro4Kjkn"
access_token = "3228323528-sbgExLXrB0ccX93cdIw4dzO0PslFxc7AZkxujUK"
access_secret = "daMaz0AhjqTJ9uQzbmWZkCcXUlswueETfFxD205744WQI"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        print(all_data)
        tweet = all_data["text"]
        print(tweet)
        # print(data)
        return(True)

    def on_error(self, status):
        print(status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["warhammer"])

