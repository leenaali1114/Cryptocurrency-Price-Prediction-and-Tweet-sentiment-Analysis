from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels
from nltk.classify import ClassifierI
from statistics import mode
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from flask import render_template, request, session, redirect, url_for, flash

import tweepy
from apikeys import *

app = Flask(__name__)

app.secret_key = "49ee3ae1f3a1ec509107deec4f4acd805383522269b9674a4b7e205e0c3b96f1"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///info.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', password='{self.password}')>"
    

class Price(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    price = db.Column(db.String(50))
    market_cap = db.Column(db.String(50))
    volume_24h = db.Column(db.String(50))
    supply = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Price(name='{self.name}', price='{self.price}', market_cap='{self.market_cap}', volume_24h='{self.volume_24h}', supply='{self.supply}', timestamp='{self.timestamp}')>"



word_feature_path = open('../models/word_features.pickle', 'rb')
word_features = pickle.load(word_feature_path)
word_feature_path.close()

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
         features[w] = (w in words)
    return features

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

client = tweepy.Client(bearer_token)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/bitcoin', methods=['GET', 'POST'])
def bitcoin_page():
    bitcoin_price = scrape_bitcoin_price()
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/bitcoin_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/bitcoin_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)
        
        
    

        return render_template('bitcoin.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred, bitcoin_price=bitcoin_price)


    return render_template('bitcoin.html')

@app.route('/ethereum', methods=['GET', 'POST'])    
def ethereum_page():
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/ethereum_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/ethereum_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)
        
        return render_template('ethereum.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred)

    return render_template('ethereum.html')

@app.route('/litecoin', methods=['GET', 'POST'])
def litecoin_page():
    if request.method == 'POST':
        time_frame = request.form.get('time_frame')
        start_date = request.form.get('start')
        end_date = request.form.get('end')
        if(time_frame == 'daily'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='d')
            pickle_in = open("../models/litecoin_daily.pkl", 'rb')
            daily_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = daily_model.predict(start=start_date, end=end_date)
        elif(time_frame == 'monthly'):
            dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),freq='m')
            pickle_in = open("../models/litecoin_monthly.pkl", 'rb')
            monthly_model = pickle.load(pickle_in)
            pickle_in.close()
            pred = monthly_model.predict(start=start_date, end=end_date)

        return render_template('litecoin.html', check=True ,start_date = start_date, end_date = end_date, labels = dates, predictions = pred)


    return render_template('litecoin.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_page():
    if request.method == 'POST':
        pickle_in = open("../models/naivebayes_final.pickle", 'rb')
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("../models/MNB_classifier_final.pickle", 'rb')
        MNB_classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open("../models/bernoulliNB_classifier_final.pickle", 'rb')
        BernoulliNB_classifier = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("../models/LogisticRegression_classifier_final.pickle", 'rb')
        LogisticRegression_classifier = pickle.load(pickle_in)
        pickle_in.close()

        voted_classifier = VoteClassifier(classifier,
                                MNB_classifier, 
                                BernoulliNB_classifier, 
                                LogisticRegression_classifier)
        
        form_name = request.form['form-name']
        if (form_name == 'form1'):
            tweet = request.form.get('tweet')
            feature_set = find_features(tweet)
            classification = [voted_classifier.classify(feature_set), voted_classifier.confidence(feature_set)]
            return render_template('sentiment.html', check = 2, classification = classification)
        if (form_name == 'form2'):
            authenticator = tweepy.OAuthHandler(api_key, api_key_secret)
            authenticator.set_access_token(access_token, access_token_secret)
            api = tweepy.API(authenticator, wait_on_rate_limit=True)
            currency = request.form.get('currency')
            query = currency + ' lang:en'
            #crypto_coin = "bitcoin"
            search_term = f'{currency} -filter:retweets'
            tweet_cursor = tweepy.Cursor(api.search_tweets, q= search_term, lang="en", tweet_mode="extended").items(100)
            
            tweetss = [tweet.full_text for tweet in tweet_cursor]
            tweets = client.search_recent_tweets(query=query, max_results=100)
            print(tweets)
            
            X = []
            Y = []
            x  = 0
            y = 0
            for tweet in tweets.data:
                x += 1
                raw_tweet = tweet.text
                clean_tweet = raw_tweet.replace("\n", "")
                feature_set = find_features(clean_tweet)
                classification = voted_classifier.classify(feature_set)
                if(classification == 'positive'):
                    y += 1
                elif(classification == 'negative'):
                    y -= 1
                else:
                    continue
                X.append(x)
                Y.append(y)
            return render_template('sentiment.html', check=1, X = X, Y = Y, currency = currency)
    return render_template('sentiment.html')

def scrape_and_store_prices():
    url = 'https://www.blockchain.com/explorer/prices'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        flash(f"An error occurred while requesting the URL: {e}","danger")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    name = soup.find_all(class_="sc-89fc2ff1-5 fYsYrO")
    price = soup.find_all(class_="sc-89fc2ff1-0 iQXnyB")
    market_cap = soup.find_all(class_="sc-89fc2ff1-11 cBoudl")
    volume_24h = soup.find_all(class_="sc-89fc2ff1-16 jBxFfE")
    supply = soup.find_all(class_="sc-89fc2ff1-17 pyRes")

    parsed_info = []
    
    for names, prices, market_caps, volumes, supplies in zip(name, price, market_cap, volume_24h, supply):
        parsed_info.append((names.text.strip(), prices.text.strip(), market_caps.text.strip(), volumes.text.strip(), supplies.text.strip()))

    try:
        # Comment the line below if database is not working for some reason and uncomment it after first parsing
        Price.query.delete()

        db.create_all()
        for info in parsed_info:
            price = Price(name=info[0], price=info[1], market_cap=info[2], volume_24h=info[3], supply=info[4])
            db.session.add(price)
        
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"An error occurred while storing the data: {e}","danger")

def scrape_bitcoin_price():
    url = 'https://www.blockchain.com/explorer/prices'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        flash(f"An error occurred while requesting the URL: {e}","danger")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    price = soup.find(class_="sc-89fc2ff1-0 iQXnyB")

    if price:
        return price.text.strip()
    else:
        return None

@app.route('/prices')
def prices():
    scrape_and_store_prices()
    prices = Price.query.all()
    return render_template('prices.html', prices=prices)


@app.route('/bitcoin')
def bitcoin():
    bitcoin_price = scrape_bitcoin_price()
    return render_template('bitcoin.html', bitcoin_price=bitcoin_price)

@app.route('/exchange', methods=['GET', 'POST'])
def exchange():
    if request.method == 'POST':
        bitcoin_amount = float(request.form['bitcoin_amount'])
        bitcoin_price = scrape_bitcoin_price()

        if bitcoin_price:
            bitcoin_price = float(bitcoin_price.replace("$", "").replace(",",""))
            exchanged_amount = bitcoin_amount * bitcoin_price
            return render_template('exchange.html', bitcoin_price=bitcoin_price, bitcoin_amount=bitcoin_amount, exchanged_amount=exchanged_amount)
        else:
            return render_template('exchange.html', bitcoin_price=None)

    return render_template('exchange.html', bitcoin_price=None)




if __name__ == '__main__':
    app.run(debug=True)