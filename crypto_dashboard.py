import streamlit as st
import requests
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv



from datetime import datetime
import matplotlib.pyplot as plt

import snscrape.modules.twitter as sntwitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer




# --- Function to get current price ---
def get_crypto_price(coin_id='bitcoin'):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url)
        data = response.json()
        return data[coin_id]['usd']
    except KeyError:
        return "Error"
    except Exception as e:
        return f"Error: {str(e)}"

# --- ✅ Function to get historical prices (last 7 days) ---
def get_historical_prices(coin_id='bitcoin', days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    
    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['date', 'price']]


def get_crypto_news(coin_symbol='BTC', api_key=os.getenv("crypto_api_key")):
    url = f"https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        'auth_token': api_key,
        'currencies': coin_symbol,
        'public': 'true'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'results' in data:
        return [post['title'] for post in data['results']]
    else:
        return ["No news found."]


def analyze_sentiment_vader(text_list):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}
    
    for text in text_list:
        score = analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            sentiment_scores["positive"] += 1
        elif score['compound'] <= -0.05:
            sentiment_scores["negative"] += 1
        else:
            sentiment_scores["neutral"] += 1
            
    return sentiment_scores




# --- ✅ Streamlit App UI ---
st.set_page_config(page_title="Crypto Sentiment & Price Dashboard")
st.title("📊 Crypto Sentiment & Price Dashboard")
st.markdown("This dashboard shows real-time crypto prices and sentiment analysis from news sources.")


# Dropdown to select coin
coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "Solana": "solana"
}
selected_coin = st.selectbox("Select a cryptocurrency", list(coins.keys()))

# Get and show price
price = get_crypto_price(coins[selected_coin])

if isinstance(price, str) and "Error" in price:
    st.error(f"❌ Failed to fetch price: {price}")
else:
    st.metric(label=f"💰 {selected_coin} Price (USD)", value=f"${price}")


# --- ✅ Show 7-Day Price Chart ---
st.subheader(f"📉 {selected_coin} Price Trend - Last 7 Days")
df_price = get_historical_prices(coins[selected_coin], days=7)
st.line_chart(df_price.set_index('date')['price'])





# --- News-Based Sentiment Analysis ---
st.subheader(f"📰 News Sentiment for {selected_coin}")

symbol_map = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Dogecoin": "DOGE",
    "Solana": "SOL"
}
news = get_crypto_news(symbol_map[selected_coin])
sentiment = analyze_sentiment_vader(news)

st.write("📊 Sentiment Breakdown from News:")
st.write(sentiment)

# Show pie chart
fig, ax = plt.subplots()
ax.pie(sentiment.values(), labels=sentiment.keys(), autopct='%1.1f%%', colors=['green', 'gray', 'red'])
ax.axis('equal')
st.pyplot(fig)

# Optional: show sample headlines
st.write("🗞️ Recent Headlines:")
for headline in news[:5]:
    st.markdown(f"- {headline}")


# ai summary
client = Groq(api_key=os.getenv("groq_api_key"));

def generate_ai_crypto_summary(coin_name, current_price, news_sentiment, price_df):
    price_summary = "\n".join(
        [f"{row['date'].strftime('%Y-%m-%d')}: ${row['price']:.2f}" for _, row in price_df.iterrows()]
    )

    prompt = f"""
You are an expert crypto financial advisor. Analyze the following data and give a summary and recommendation for "{coin_name}".

Current Price: ${current_price}

News Sentiment:
- Positive: {news_sentiment.get("positive", 0)}
- Neutral: {news_sentiment.get("neutral", 0)}
- Negative: {news_sentiment.get("negative", 0)}

Price Trend (Last 7 Days):
{price_summary}

Output must include:
1. Sentiment overview.
2. Price trend summary.
3. Recommendation: Buy / Hold / Sell
4. In last just tell in brief wheteher to buy , sell or hold of future scope of that coin
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content



st.subheader("🤖 AI Investment Recommendation")

ai_summary = generate_ai_crypto_summary(
    selected_coin,
    price,
    sentiment,
    df_price
)

st.markdown(ai_summary)
