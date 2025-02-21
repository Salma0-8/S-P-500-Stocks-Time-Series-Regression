# 📈 S&P 500 Stocks Time Series Regression  

## 🚀 Introduction  
Hey there! 👋 I'm diving deep into the world of **S&P 500 stock market trends**, building a **time series regression model** to predict stock movements. This project is a mix of **data exploration, feature engineering, and machine learning** to gain valuable financial insights.  

## 📊 Data Overview  
I've got **three datasets** that power this analysis:  

### **1️⃣ S&P 500 Companies (`sp500_companies.csv`)**  
This dataset contains details about **S&P 500 companies**, including their **sector, industry, market cap, financials, and stock weights**.  

🔍 **Initial Findings:**  
- The big tech giants dominate: **Apple, NVIDIA, Microsoft, and Alphabet** hold the highest market cap.  
- **NVIDIA** is a growth beast, showing **122.4% revenue growth**, outpacing others.  
- **Apple has the highest weight (0.0635)** in the index, meaning it significantly impacts overall movements.  

### **2️⃣ S&P 500 Index (`sp500_index.csv`)**  
This dataset tracks the **historical performance of the S&P 500** index.  

📌 **First few entries:**  
- The **earliest date is October 6, 2014**, with the index at **1,964.82 points**.  
- The **daily fluctuations** reflect economic trends, earnings reports, and investor sentiment.  

💡 **Next step:** Plot the **S&P 500 index over time** to see the long-term trend.  

### **3️⃣ S&P 500 Stock Prices (`sp500_stocks.csv`)**  
This dataset contains **historical stock prices** for every S&P 500 company.  

🔍 **Example (3M - MMM on Jan 4, 2010):**  
- **Open:** 69.47, **Close:** 69.41  
- **High:** 69.77, **Low:** 69.12  
- **Volume:** 3.64M shares traded  

📌 **Why this matters:**  
- The **"Adj Close"** column accounts for dividends/splits, making it crucial for **accurate return calculations**.  

Here's how I'd write this step in **GitHub README.md** format while keeping the tone like yours:  

---

## 🛠️ Step 2: Data Preprocessing  

Before diving into analysis, I cleaned and prepped the data to make it **ready for modeling**. Here’s what I did:  

### 🔄 **Convert Date Columns to Datetime Format**  
Since I'm working with **time series data**, it's crucial to ensure that dates are in the proper format.  

```python
sp500_index['Date'] = pd.to_datetime(sp500_index['Date'])
sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])
```

### 📌 **Sort Data by Date**  
Sorting helps in **sequential analysis and forecasting**.  

```python
sp500_index = sp500_index.sort_values('Date')
sp500_stocks = sp500_stocks.sort_values('Date')
```

### 🚨 **Handle Missing Values**  
Missing values can mess up ML models, so I filled them with **zero (0)** for now.  

```python
sp500_companies.fillna(0, inplace=True)
sp500_stocks.fillna(0, inplace=True)
```

### 📊 **Set the Index for Time Series Analysis**  
Setting the **'Date'** column as an index makes it easier to visualize and apply time series models.  

```python
sp500_index.set_index('Date', inplace=True)
```

✅ **Now the data is clean and ready for exploration!**

## 📊 Step 3: Exploratory Data Analysis (EDA)

### 📈 S&P 500 Index Trend
To understand the historical performance of the **S&P 500 Index**, I plotted the trend over time.

![S&P 500 Index Trend](SP%20500%20index%20trend.png)

### 📉 AAPL Stock Price Trend
I also analyzed **Apple's (AAPL) stock price movements** over time.

![AAPL Stock Price Trend](SP%20appl%20stock.png)

### 🧑‍💻 Code Snippets
Below are the Python code snippets used for visualization:

```python
# S&P 500 Index Trend
plt.figure(figsize=(12, 6))
plt.plot(sp500_index.index, sp500_index['S&P500'], label="S&P 500 Index")
plt.title("S&P 500 Index Trend")
plt.xlabel('Year')
plt.ylabel('Index Value')
plt.legend()
plt.show()

# AAPL Stock Price Trend
aapl_stock = sp500_stocks[sp500_stocks['Symbol'] == 'AAPL']
plt.figure(figsize=(12, 6))
plt.plot(aapl_stock['Date'], aapl_stock['Close'], label='AAPL Close Price')
plt.title('AAPL Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

