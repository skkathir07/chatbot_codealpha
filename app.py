from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample 40 AI FAQs
faqs = [
    {"question": "What is Artificial Intelligence?", "answer": "AI is the simulation of human intelligence in machines."},
    {"question": "Define Machine Learning.", "answer": "Machine Learning is a subset of AI that allows systems to learn from data."},
    {"question": "What is Deep Learning?", "answer": "Deep Learning is a subset of ML using neural networks with many layers."},
    {"question": "Define Neural Networks.", "answer": "Neural Networks are computing systems inspired by the human brain."},
    {"question": "What is NLP?", "answer": "Natural Language Processing enables computers to understand human language."},
    {"question": "What is Computer Vision?", "answer": "Computer Vision allows computers to interpret visual information from the world."},
    {"question": "Define Supervised Learning.", "answer": "Supervised learning trains models using labeled data."},
    {"question": "Define Unsupervised Learning.", "answer": "Unsupervised learning finds patterns in unlabeled data."},
    {"question": "What is Reinforcement Learning?", "answer": "An agent learns by interacting with an environment to maximize rewards."},
    {"question": "What is a Dataset?", "answer": "A dataset is a collection of data used for training or testing models."},
    {"question": "Define Overfitting.", "answer": "Overfitting is when a model learns training data too well and fails on new data."},
    {"question": "Define Underfitting.", "answer": "Underfitting is when a model fails to capture data patterns."},
    {"question": "What is Transfer Learning?", "answer": "Using a pre-trained model on a new similar problem."},
    {"question": "What are Hyperparameters?", "answer": "Configurations set before training a model."},
    {"question": "Define Epoch.", "answer": "One complete pass through the entire dataset during training."},
    {"question": "What is Gradient Descent?", "answer": "An optimization algorithm to minimize loss function."},
    {"question": "Define Loss Function.", "answer": "It measures how well the model predicts compared to actual values."},
    {"question": "What is Activation Function?", "answer": "Introduces non-linearity in neural networks."},
    {"question": "Name common Activation Functions.", "answer": "ReLU, Sigmoid, Tanh."},
    {"question": "What is Backpropagation?", "answer": "Algorithm to adjust weights in neural networks by propagating errors backward."},
    {"question": "Define Precision.", "answer": "True Positives divided by total predicted positives."},
    {"question": "Define Recall.", "answer": "True Positives divided by total actual positives."},
    {"question": "Define F1 Score.", "answer": "Harmonic mean of precision and recall."},
    {"question": "What is CNN?", "answer": "Convolutional Neural Networks are used for image data."},
    {"question": "What is RNN?", "answer": "Recurrent Neural Networks are used for sequential data."},
    {"question": "Define GAN.", "answer": "Generative Adversarial Networks generate new data samples."},
    {"question": "What is Tokenization?", "answer": "Breaking text into words or tokens in NLP."},
    {"question": "What is Lemmatization?", "answer": "Converting words to their base form in NLP."},
    {"question": "Define Clustering.", "answer": "Grouping data points based on similarity without labels."},
    {"question": "What is K-Means?", "answer": "A clustering algorithm to partition data into K clusters."},
    {"question": "What is Decision Tree?", "answer": "A model that splits data into branches to make predictions."},
    {"question": "Define Random Forest.", "answer": "An ensemble of decision trees for better accuracy."},
    {"question": "What is SVM?", "answer": "Support Vector Machine is used for classification tasks."},
    {"question": "Define Naive Bayes.", "answer": "A probabilistic classifier based on Bayes’ theorem."},
    {"question": "What is Bias-Variance Tradeoff?", "answer": "Balancing model simplicity (bias) and complexity (variance)." },
    {"question": "What is Dimensionality Reduction?", "answer": "Reducing input variables to simplify models (e.g., PCA)."},
    {"question": "What is PCA?", "answer": "Principal Component Analysis reduces data dimensions keeping max variance."},
    {"question": "Define Feature Engineering.", "answer": "Creating new features to improve model performance."},
    {"question": "What is Cross-validation?", "answer": "Technique to evaluate models on different subsets of data."},
    {"question": "Define Confusion Matrix.", "answer": "A table showing correct and incorrect predictions in classification."}
]

# Prepare vectorizer
questions = [faq["question"] for faq in faqs]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["message"]
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    response = faqs[index]["answer"]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
