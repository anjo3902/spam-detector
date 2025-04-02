# Email Spam Detection using Machine Learning

## Overview
This is an **Email Spam Detection App** built using **Streamlit** and **Machine Learning**. The app allows users to enter an email and determine whether it is **Spam** or **Not Spam** using different classifiers.

## Features
- **Classifiers Supported:**
  - Logistic Regression
  - Naïve Bayes
  - Decision Tree
- **Natural Language Processing (NLP) Techniques:**
  - Tokenization
  - Stopword Removal
  - Lemmatization
  - TF-IDF Vectorization
- **Simple Web Interface using Streamlit**
- **Deployable on Streamlit Cloud**

---

## Project Structure
```
email-spam-detection/
│── model/                   # Trained models
│── data/                    # Dataset (if applicable)
│── app.py                   # Streamlit app script
│── train.py                 # Script for training models
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── saved_model.pkl          # Saved ML model
```

---

## Installation & Setup

### ** 1️ Clone the Repository**
```sh
git clone https://github.com/YOUR_USERNAME/email-spam-detector.git
cd email-spam-detector
```

### ** 2️ Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

### ** 3️ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## Training the Model
To train the spam detection model, run:
```sh
python train.py
```
This script processes the dataset, applies NLP techniques, and trains the selected machine learning models.

---

## Running the Streamlit App
Start the Streamlit web app using:
```sh
streamlit run app.py
```

This will launch the app in your default browser.

---

## Deployment on Streamlit Cloud
To deploy the app on **Streamlit Cloud**, follow these steps:

1. Push the code to **GitHub**.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Click **New App** → Select your **GitHub Repo**.
4. Set the **main script path** as `app.py`.
5. Deploy!

Once deployed, you’ll get a public link like:
```
https://anjo3902-spam-detector-app-rd1zah.streamlit.app/
```

---

## Usage Guide
1. Open the **Streamlit App**.
2. Enter the **email text** in the input field.
3. Select a **Machine Learning classifier**.
4. Click on **Predict** to check if the email is spam or not.
5. The result will be displayed instantly.

---

## Technologies Used
- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **Numpy**
- **NLTK**
- **TfidfVectorizer**

---

## Contact
 Email: anjojaison3902@gmail.com  
 LinkedIn: www.linkedin.com/in/anjo-jaison-p-b0373a249 

---

## Contribute
If you'd like to contribute, fork the repo and submit a **pull request**. Any contributions are welcome!

---

## License
This project is **open-source** under the **MIT License**.

---

### Star this repository if you like it!

