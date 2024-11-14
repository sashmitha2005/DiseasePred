from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load and split data
data = pd.read_csv('Training.csv')
X = data.drop('prognosis', axis=1)
y = data['prognosis']
symptom_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers and pipelines
pipelines = {
    'RandomForest': Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('classifier', SVC(probability=True))
    ]),
    'KNN': Pipeline([
        ('classifier', KNeighborsClassifier())
    ]),
    'NaiveBayes': Pipeline([
        ('classifier', GaussianNB())
    ]),
}

# Define hyperparameter grids for tuning
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 15, 20],
        'classifier__min_samples_split': [2, 4, 6]
    },
    'SVM': {
        'classifier__C': [0.1, 0.5, 1],
        'classifier__gamma': ['scale', 'auto']
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    }
}

# Tune each classifier using GridSearchCV
best_estimators = {}
for name, pipeline in pipelines.items():
    if name in param_grids:
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimators[name] = grid_search.best_estimator_
        print(f"{name} best parameters: {grid_search.best_params_}")
        print(f"{name} best cross-validated accuracy: {grid_search.best_score_:.2f}")
    else:
        pipeline.fit(X_train, y_train)
        best_estimators[name] = pipeline

# Voting Classifier with best estimators
voting_clf = VotingClassifier(estimators=[
    ('RandomForest', best_estimators['RandomForest']),
    ('SVM', best_estimators['SVM']),
    ('KNN', best_estimators['KNN']),
    ('NaiveBayes', best_estimators['NaiveBayes']),
], voting='hard')
voting_clf.fit(X_train, y_train)

# Set up Flask app
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
    
    for symptom in symptoms:
        if symptom in symptom_columns:
            index = symptom_columns.index(symptom)
            input_vector[index] = 1
        else:
            return jsonify({"error": f"Symptom '{symptom}' not recognized"}), 400
    
    symptoms_reshaped = [input_vector]
    prediction = voting_clf.predict(symptoms_reshaped)
    return jsonify({'predicted_disease': prediction[0]})


if __name__ == '__main__':
    app.run(debug=False)
