from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('Training.csv')
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# List of symptom columns
symptom_columns = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=4, random_state=42),
    'SVM': SVC(C=0.5, gamma='scale', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'NaiveBayes': GaussianNB(),
}

# Train classifiers and evaluate accuracy
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.2f}")

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('RandomForest', classifiers['RandomForest']),
    ('SVM', classifiers['SVM']),
    ('KNN', classifiers['KNN']),
    ('NaiveBayes', classifiers['NaiveBayes']),
], voting='hard')

voting_clf.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/redirect_to_index')
def redirect_to_index():
    return redirect(url_for('index'))

@app.route('/Description')
def description():
    return render_template('Description.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    input_vector = [0] * len(symptom_columns)
    
    # Map symptoms to input_vector
    for symptom in symptoms:
        if symptom in symptom_columns:
            index = symptom_columns.index(symptom)
            input_vector[index] = 1
        else:
            return jsonify({"error": f"Symptom '{symptom}' not recognized"}), 400
    
    symptoms_reshaped = [input_vector]
    prediction = voting_clf.predict(symptoms_reshaped)
    return jsonify({'predicted_disease': prediction[0]})

# Run the app
def handler(request):
    return app(request)

if __name__ == '__main__':
    app.run(debug=False)
