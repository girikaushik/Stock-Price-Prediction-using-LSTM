# 📈 LSTM Stock Price Prediction

A deep learning project that uses **Long Short-Term Memory (LSTM)** networks to predict stock prices based on historical price data.

The model learns patterns from the previous **50 trading days** and attempts to predict the stock's price for the next day.

> ⚠️ **Disclaimer:** This project is for educational and experimentation purposes only. Stock-market predictions are inherently uncertain and this model should not be used as financial advice.

---

## 🧠 Project Overview

This project implements a time-series forecasting pipeline using:

- **Python**
- **TensorFlow / Keras**
- **LSTM**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**

The model uses the **High** price from historical stock data.

### Prediction Approach

```text
Historical Stock Data
        ↓
Select High Price
        ↓
Train/Test Split (80/20)
        ↓
Min-Max Scaling
        ↓
Create 50-Day Sequences
        ↓
     LSTM Model
        ↓
Predict Next Day
        ↓
Inverse Scaling
        ↓
Compare with Actual Price
```

---

## 🏗️ Model Architecture

The neural network consists of **three LSTM layers** followed by a Dense output layer.

```text
Input
  │
  │  50 previous trading days
  ▼
LSTM (90 units)
  │
Dropout (20%)
  ▼
LSTM (90 units)
  │
Dropout (20%)
  ▼
LSTM (90 units)
  │
Dropout (20%)
  ▼
Dense (1)
  │
  ▼
Predicted Stock Price
```

### Model Configuration

| Parameter | Value |
|---|---:|
| Lookback window | 50 days |
| Training data | 80% |
| Testing data | 20% |
| LSTM layers | 3 |
| LSTM units | 90 |
| Dropout | 20% |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Epochs | 50 |
| Batch size | 32 |

---

## 🔬 Data Preprocessing

The historical dataset is loaded using Pandas:

```python
df = pd.read_csv("REL(2).csv")
```

The project currently uses the **High** column:

```python
ex = df["High"].values
```

The data is then normalized using `MinMaxScaler`:

```text
Original values
      ↓
Min-Max Scaling
      ↓
Values between 0 and 1
```

This helps the LSTM network train more effectively.

---

## ⏳ Sequence Generation

The model uses a **50-day lookback window**.

For example:

```text
Days 1–50   → Predict Day 51
Days 2–51   → Predict Day 52
Days 3–52   → Predict Day 53
...
```

The `create_my_dataset()` function converts the time series into input/output sequences.

### Input

```text
50 historical prices
```

### Output

```text
Next day's price
```

This creates a supervised learning problem from the original time-series data.

---

## 🧪 Training

The model is trained using:

```python
model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32
)
```

The trained model is saved as:

```text
stock_prediction.h5
```

---

## 📊 Prediction & Visualization

After training, the model predicts prices for the test dataset.

The predictions are converted back from the normalized scale using:

```python
scaler.inverse_transform(predictions)
```

The project generates visualizations comparing:

- 📈 Original stock prices
- 🔵 Model predictions
- 🔴 Actual test prices

This provides a visual understanding of how closely the LSTM follows the historical price movement.

---

## 📁 Project Structure

```text
LSTM-Stock-Prediction/
│
├── tradify2.ipynb
├── REL(2).csv
├── stock_prediction.h5
└── README.md
```

> Dataset and trained model files can be excluded from GitHub if they are large. A small sample dataset can be provided instead.

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Then open the notebook:

```bash
jupyter notebook tradify2.ipynb
```

Or run it using Google Colab.

---

## 🚀 How to Run

1. Prepare your historical stock dataset.
2. Update the CSV path in the notebook.
3. Make sure the dataset contains a `High` column.
4. Run the preprocessing cells.
5. Generate the training sequences.
6. Train the LSTM model.
7. Generate predictions.
8. Visualize predicted vs actual prices.

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Time-series forecasting
- Sequence generation
- Feature scaling
- LSTM neural networks
- Dropout regularization
- Train/test splitting
- Model persistence
- Prediction visualization
- TensorFlow/Keras

---

## 🔮 Possible Improvements

Future versions could improve
