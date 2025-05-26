from flask import Flask, render_template, request
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    ticker = request.form['ticker'].upper()

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty:
            return render_template('index.html', error=f"No data found for '{ticker}'.")

        hist.reset_index(inplace=True)
        hist['DayIndex'] = np.arange(len(hist))

        recent = hist.tail(90)
        X = recent[['DayIndex']]
        y = recent['Close']
        model = LinearRegression()
        model.fit(X, y)

        future_days = 30
        last_day_index = hist['DayIndex'].iloc[-1]
        future_indices = np.arange(last_day_index + 1, last_day_index + future_days + 1).reshape(-1, 1)
        future_prices = model.predict(future_indices)

        last_date = hist['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

        combined_dates = list(hist['Date']) + future_dates
        combined_prices = list(hist['Close']) + list(future_prices)

        plt.figure(figsize=(10, 5))
        plt.plot(combined_dates, combined_prices, color='blue', label='Past + Predicted')
        plt.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
        plt.title(f"Stock Forecast for {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('result.html', ticker=ticker, image_file='plot.png')

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
