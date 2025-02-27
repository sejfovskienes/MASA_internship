1. Collecting Data
We tried scraping data with our own script, but then we found a dataset that contained the data we needed. The dataset we are using can be downloaded from the following link: https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction

2. Dataset
The dataset (stock_tweets.csv) consists of 4 columns:
•	Date (in format '%Y-%m-%d %H:%M:%S%z')
•	Tweet (the content of the tweet)
•	Stock Name (the stock ticker)
•	Company Name (the full company name)

3. Reformatting Dataset
In order to use the dataset in further training of models, we needed to make a few adjustments. After running a short script on the original stock_tweets.csv file, the new reformatted_stock_tweets.csv file consists of the following 6 columns:
•	Number of row
•	Date (in format '%m/%d/%Y')
•	Tweet (same as in stock_tweets.csv)
•	Stock Name (same as in stock_tweets.csv)
•	Company Name (same as in stock_tweets.csv)
•	Close (this is a newly scraped column that represents the closing price of the stock for that date. It has been scraped using the Yahoo Finance API)

3.1. Inspecting nulls
There were some null values in the Close column. After manually checking the dates we came to conclusion that the dates where data is missing are weekends (when the stock market is closed). To address this, we used the ForwardFill method to propagate the last available value forward and fill the gaps.

4. DistilRoBERTa Transformer
In order to predict the sentiment of the tweets, we used a pre-trained sentiment analysis model from the Hugging Face library, specifically the DistilRoBERTa model fine-tuned for financial news sentiment analysis. The code applies this model to each tweet in the Tweet column of the dataset, extracting the sentiment label ("positive", "negative" or "neutral") for each tweet. These predictions are stored in a new column called Predictions. Finally, the updated dataset with sentiment predictions is saved as a CSV file (predicted.csv) for further analysis.

5. XGB_model
In this section we will be trying to predict the closing price of the stocks using time series analysis.
To create features for time series analysis, we generated lagged versions of the Close column by shifting its values by 1 to 5 days. Since shifting introduces missing values in the first few rows, we removed them.
We extracted the Month and Year columns from the Date column and applied cyclical encoding using sine and cosine transformations. This helps represent the periodic nature of time, ensuring that the model we will be using understands that January is after December. After encoding, we remove the original Date, Month and Year columns, leaving only the transformed features Month_Sin, Month_Cos, Year_Sin and Year_Cos.
Because we alredy did sentiment analysis on the tweets, we removed the Tweet column.
We used Label Encoding to convert Stock Name and Company Name into numerical values, and then normalized them using MinMax Scaling. 
The prices in the Close column are also scaled in range from 0 to 1 range. 
The sentiment column Category is transformed into numerical values, where positive = 1, negative = -1, and neutral = 0.
We split the dataset into training and testing sets. We set 80% of the total dataset as training set and 20% as testing set.
We used XGBoost regression model to predict the stock prices. It initializes an XGBRegressor with the following settings:
•	objective='reg:squarederror'
•	n_estimators=100 
•	learning_rate=0.1 
•	max_depth=3 
•	random_state=42
•	booster='gblinear'

We got the following results:
--------------------------------------------------------------------------------------------------------------------------------------
Mean squared error of the Close prediction is: 3552.5207878007577
Mean absolute error of the Close prediction is: 56.74709877763207
R2 score of the Close prediction is: 0.49342609244693947
--------------------------------------------------------------------------------------------------------------------------------------
 
The results below are from training the same model without lags using train_test_split
--------------------------------------------------------------------------------------------------------------------------------------
Mean squared error of the Close prediction is: 35796.08135728018
Mean absolute error of the Close prediction is: 110.40202184618431
R2 score of the Close prediction is: -4.104642438122249
--------------------------------------------------------------------------------------------------------------------------------------

The results below are from training the same model without lags using TimeSeriesSplit object
--------------------------------------------------------------------------------------------------------------------------------------
Average R2_score of all splits: 0.6385370679400936
--------------------------------------------------------------------------------------------------------------------------------------

The results below are from training the same model without lags using TimeSeriesSplit object
--------------------------------------------------------------------------------------------------------------------------------------
Average R2_score of all splits: -1.5482989543091554
--------------------------------------------------------------------------------------------------------------------------------------
