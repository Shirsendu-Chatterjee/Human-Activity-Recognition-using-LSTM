# Human-Activity-Recognition-using-LSTM
This project uses a multi-layered **LSTM neural network** to classify human physical activities (like walking, standing, sitting) based on motion sensor data from smartphones. The dataset is sourced from the **UCI HAR dataset** available on Kaggle.

---

## 📊 Dataset

- **Name:** UCI Human Activity Recognition (HAR) Dataset  
- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)  
- **Classes (6):**
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying

- **Input Shape:** `(samples, 128, 9)`  
  *(128 timesteps, 9 sensor signals)*

---

## 🧠 Model Architecture

The model is built using **Keras Sequential API** with LSTM layers:

```python
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(128, 9), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(5, activation='relu', return_sequences=False))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))  # 6 output classes
