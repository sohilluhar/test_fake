import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


class analysis1:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def dd(self,searchTerm):

        consumerKey ='gPVcbgDVwhQJKnK5R71hVcgBm'# 'TybsmmSJzoO7E5aF6pLyYRwGY'
        consumerSecret = 'wYXjNGqlebXbW0FRPBNxWAUoxF118oZFJ33Ooyd6p0aFF6tQCC'
        accessToken = '721717156694470657-qEEtuThViYit6lRBA3kD5mKx3IsJMx6'#'2686897252-dWVqehXZvvOi2IQmyqJUEhG3gK2XuMsilJgyVG2'
        accessTokenSecret = 'nUyeIu0ZJldPdLP9bBhcC6JLMduq1kDOnwIfH3IMAbSIV'#'agwFQZXs4FWlvl8B41oOtQQuq8j3T9NZAoxuNCPSgAD1G'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

       # searchTerm = input("Enter Keyword/Tag to search about: ")
        NoOfTerms = 10


        self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
        analysis=""
      


        for tweet in self.tweets:
            #Normal tweets without clean

            print(tweet.text)


            #Tweets with clean process
            print(self.cleanTweet(tweet.text))


            #Tweets with only one tweets
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            analysis = TextBlob(tweet.text)
        
            

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing  tweets.")
        return analysis if analysis!=""  else ""

    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())
