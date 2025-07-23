# ✈️ Airline Passenger Forecasting with Recurrent Neural Networks (RNN)

This project demonstrates time series forecasting of monthly international airline passengers using Recurrent Neural Networks (RNN). It utilizes the classic **Airline Passengers Dataset** (1949–1960) to train and evaluate a deep learning model for predicting future values.

---

## 📊 Project Overview

- **Goal**: Forecast future monthly airline passenger counts using an RNN-based model.
- **Dataset**: [Airline Passengers Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
- **Model**: Simple RNN / LSTM implemented using TensorFlow/Keras.
- **Tools**: Python, Pandas, Matplotlib, scikit-learn, TensorFlow/Keras

---

## 🧠 Problem Statement

The dataset contains historical monthly totals of international airline passengers from 1949 to 1960. The objective is to:
- Preprocess and visualize the data
- Transform the time series into a supervised learning format
- Train an RNN-based model for multi-step forecasting
- Evaluate prediction accuracy

---

## 📁 Project Structure


---

## 📦 Dependencies

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow

🧪 Key Steps in Notebook
1. Data Loading & Exploration
Dataset sourced from Brownlee's GitHub.

Inspected for missing values, date formatting, and trends.

2. Preprocessing
Normalized values using MinMaxScaler

Converted time series into a supervised learning problem using a sliding window

3. Model Architecture
Implemented an RNN or LSTM model using TensorFlow/Keras.

Configured layers, loss function, and optimizer.

Trained on sequences to predict next time step.

4. Evaluation & Forecasting
Forecasted values plotted alongside actuals.

RMSE or MAE calculated for performance.

📈 Example Forecast Plot
(Insert a chart image or a placeholder here if using GitHub Pages or image folder)

📌 Results
The RNN model was able to capture the upward trend and seasonality.

Simple LSTM/RNN models work well for univariate time series like this, though performance can be improved with:

More layers or units

Bi-directional RNNs

Attention mechanisms

External regressors (e.g., holidays, economy indicators)

🔮 Future Improvements
Try GRU or deeper LSTM models

Experiment with multistep forecasting

Incorporate external factors (exogenous features)

Deploy model as a Flask API or Streamlit app

📚 References
Jason Brownlee, Time Series Forecasting with the Airline Dataset
https://github.com/jbrownlee/Datasets

TensorFlow/Keras RNN Docs:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

🧑‍💻 Author
Yusuf Olatayo Kareem
Data Science & Machine Learning Enthusiast
GitHub

📄 License
This project is open-source and available under the MIT License.