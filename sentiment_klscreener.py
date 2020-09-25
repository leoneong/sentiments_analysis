from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd

# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

i3investor_url = 'https://klse.i3investor.com/servlets/stk/nb/'

news_tables = {}
tickers = ['7155', '7247' ]

for ticker in tickers:
    url = i3investor_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response, "html.parser")
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='nbTable')
    # Add the table to our dictionary
    news_tables[ticker] = news_table

amzn = news_tables['7155']
# Get all the table rows tagged in HTML with <tr> into 'amzn_tr'
amzn_tr = amzn.findAll('tr')
amzn_tr.pop(0)
# news column
parsed_news = []

for i, table_row in enumerate(amzn_tr):
    # Read the text of the element 'a' into 'link_text'
    text = table_row.a.text
    # Read the text of the element 'td' into 'data_text'
    date = table_row.td.text

    # Append ticker, date, time and headline as a list to the 'parsed_news' list
    parsed_news.append([date, text])

# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Set column names
columns = ['date','headline']

# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

print(parsed_and_scored_news)

 
    