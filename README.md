# Project-5-Comment_Toxicity
The objective of this project is to develop a deep learning-based comment toxicity model using Python. This model will analyze text input from online comments and predict the likelihood of each comment being toxic

### Project Objective

To develop a deep learning model that automatically detects harmful or inappropriate comments and classifies them into different toxicity categories.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Deep Learning:** TensorFlow, Keras
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Development Environment:** Google Colab, VS Code

##  Dataset Description
The dataset contains six toxicity categories:

1. Toxic
2. Severe Toxic
3. Obscene
4. Threat
5. Insult
6. Identity Hate

## Project Workflow
1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Text Cleaning
4. Tokenization
5. Sequence Padding
6. Model Building
7. Model Training
8. Model Evaluation
9. Threshold Tuning
10. Streamlit Application Development

## Model Architecture
The project uses **LSTM-based Deep Learning model**.
- **Embedding Layer:** Converts numerical word representations into dense vectors.
- **LSTM Layer (64 units):** Learns sequential patterns from text.
- **Dropout Layer (0.3):** Helps prevent overfitting.
- **Dense Layer (64 units, ReLU):** Learns additional features.
- **Dropout Layer (0.3):** Further reduces overfitting.
- **Output Layer (6 units, Sigmoid):** Predicts six toxicity categories.

##  Model Performance
| Micro F1-score | 0.73 |
| Weighted F1-score | 0.74 |
| Macro F1-score | 0.58 |
| Micro Recall | 0.76 |

### Category-wise F1-score
| Toxicity Category | F1-score |
|---|---:|
| Toxic | 0.78 |
| Severe Toxic | 0.51 |
| Obscene | 0.80 |
| Threat | 0.27 |
| Insult | 0.72 |
| Identity Hate | 0.40 |

## Streamlit Application Features

### 1. Dashboard

- Displays total comments, toxic comments, and non-toxic comments.
- Shows overall model performance metrics.
- Visualizes toxicity category distribution.
- Includes sample test cases.

### 2. Single Comment Detection

- Accepts an individual comment from the user.
- Predicts whether the comment contains toxic content.
- Displays predictions for all six toxicity categories.

### 3. Bulk CSV Prediction

- Allows users to upload a CSV file.
- Processes up to **100 comments**.
- Displays toxicity prediction results.
- Provides an option to download prediction results as a CSV file.
  
## Conclusion

This project demonstrates how **Deep Learning and Natural Language Processing** can be applied to detect toxic comments.
By combining an **LSTM model** with a **Streamlit web application**, this project provides an interactive solution for identifying and classifying potentially harmful online comments.

---
