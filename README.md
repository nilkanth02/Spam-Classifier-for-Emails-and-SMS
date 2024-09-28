# Spam Classifier for Emails and SMS

## 🚀 Overview
The **Spam Classifier for Emails and SMS** is a machine learning-based solution that automatically detects and filters spam messages. This project aims to make communication cleaner by classifying incoming messages as either "spam" or "ham" (not spam). Built with Python and popular data science libraries, this classifier offers an efficient and reliable way to maintain a spam-free inbox.

## 🔍 Problem Statement
Spam messages are not only annoying but also pose significant security risks. With the increasing volume of unsolicited emails and SMS, having a system to distinguish between spam and legitimate messages is crucial. This project addresses the need for an automated, accurate solution to filter out unwanted content and minimize the risk of phishing and other cyber threats.

## 🛠️ Features
- **Real-time classification**: Quickly determines whether a message is spam or not.
- **Scalable**: Capable of handling both emails and SMS.
- **User-Friendly Interface**: Provides a simple interface for testing messages.
- **Machine Learning Model**: Uses **Naive Bayes Classifier** for its effectiveness in text classification tasks.

## 🔧 Technologies Used
- **Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, NLTK
- **Interface**: Streamlit for a clean and interactive user experience

## 📊 Dataset
The model was trained using a dataset of labeled SMS and email messages, commonly available in open datasets. Each message was labeled as either "spam" or "ham". The dataset underwent preprocessing including text cleaning, tokenization, and conversion to numerical features using **TF-IDF**.

## 🤖 Machine Learning Approach
The **Naive Bayes Classifier** was chosen due to its simplicity and high performance in text classification tasks. Key steps in the model building include:
1. **Text Preprocessing**: Removing special characters, converting to lowercase, and stopword removal.
2. **Feature Extraction**: Using **TF-IDF Vectorization** to convert text to numerical format.
3. **Model Training**: A Naive Bayes model was trained and optimized for accuracy and performance.

## 🧪 How to Run the Project
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/nilkanth02/spam-classifier.git
    cd spam-classifier
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 📈 Results
The classifier achieved an accuracy of **97%** on the test set, showcasing its effectiveness in distinguishing spam from legitimate messages. The precision, recall, and F1-score metrics indicate a well-balanced model with minimal false positives.

## 🌟 Key Learnings
- **Text Preprocessing** is critical for improving model accuracy in natural language processing.
- **Naive Bayes** is highly efficient for text classification tasks due to its probabilistic nature.
- **Feature Engineering** with TF-IDF plays an important role in transforming raw text into meaningful data that machine learning algorithms can understand.

## 💡 Future Enhancements
- **Integration with Email Clients**: Adding direct integration to classify messages within popular email clients.
- **Model Improvement**: Experimenting with more advanced models like **LSTM** or **Transformers** to improve accuracy.
- **Deployment**: Deploying the classifier as an API service for broader usability.

## 👨‍💻 Myself
- **Nilkanth Ahire** - [LinkedIn](https://linkedin.com/in/nilkanthahire) | [GitHub](https://github.com/nilkanth02)


## 🙏 Acknowledgments
- Special thanks to the creators of the open-source dataset used.
- Thanks to all contributors who have made the libraries used in this project open and free.

---

Feel free to clone, fork or contribute to make this project even better! 😄
