from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Use /tmp for Vercel (serverless environment)
TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(TEMP_DIR, 'fake_news_model.pkl')
DATASET_PATH = os.path.join(TEMP_DIR, 'news_dataset.csv')

# Global model cache
MODEL_CACHE = None

# ======================= DATA SETUP =======================

def create_synthetic_data():
    """Create synthetic dataset for fake news detection"""
    print("Creating synthetic dataset...")
    real_news = [
        "The government announced new tax reforms today to boost the economy.",
        "Scientists have discovered a new species of frog in the Amazon rainforest.",
        "The local football team won the championship after a thrilling match.",
        "NASA is planning a new mission to Mars next year.",
        "Apple released its latest iPhone model with advanced camera features.",
        "The stock market reached an all-time high amidst positive economic data.",
        "A new study shows that coffee consumption may extend lifespan.",
        "The city council approved the construction of a new park downtown.",
        "Global temperatures are rising due to climate change, experts warn.",
        "The education minister announced updates to the school curriculum.",
        "New electric vehicle charging stations are being installed across the country.",
        "Researchers found a cure for a rare genetic disease.",
        "The olympics will be held in Los Angeles in 2028.",
        "A major tech company is opening a new headquarters in the city.",
        "Local farmers report a bumper harvest this season.",
        "The prime minister visited the flood-affected areas today.",
        "A famous actor won the best actor award at the ceremony.",
        "The central bank decided to keep interest rates unchanged.",
        "A new public library was inaugurated by the mayor.",
        "The national space agency launched a weather satellite successfully.",
        "Health officials confirm new vaccine is safe and effective after clinical trials.",
        "University research shows positive results in cancer treatment trials.",
        "Economic growth reached 3.5 percent in the latest quarter report.",
        "Environmental protection act passes in parliament with bipartisan support.",
        "Major infrastructure project completed ahead of schedule.",
        "Scientists make breakthrough in renewable energy technology.",
        "International trade agreement signed by 15 nations.",
        "Hospital opens new emergency care wing.",
        "Schools report record graduation rates this year.",
        "Transportation system receives major safety upgrades."
    ]
    
    fake_news = [
        "Aliens have landed in New York and are meeting with the President!",
        "Drinking bleach can cure all known viruses instantly.",
        "The earth is actually flat and NASA is hiding the truth.",
        "Celebrity reveals secret to eternal youth: eating only dirt.",
        "Government to ban all cars by next week, sources say.",
        "Scientists prove that gravity is just a myth invented by companies.",
        "New law requires everyone to walk backwards on Tuesdays.",
        "Famous singer is actually a robot controlled by the illuminate.",
        "Shark found swimming in a cornfield in Iowa.",
        "The moon is made of cheese, confirmed by secret documents.",
        "Vegetables are toxic and should be avoided at all costs.",
        "Bigfoot was elected mayor of a small town in Oregon.",
        "The internet will be shut down permanently tomorrow.",
        "Cats are secretly planning to take over the world.",
        "Water is no longer necessary for human survival, new study claims.",
        "A man lived for 500 years by holding his breath.",
        "Money grows on trees in this hidden island.",
        "The sun is going to explode next month, panic ensues.",
        "Dinosaurs are still alive and living in the sewers.",
        "You can fly if you believe hard enough, says guru.",
        "Secret government agency controls weather with satellites.",
        "Microchips in vaccines track all human movements constantly.",
        "Celebrities are reptiles from outer space disguised as humans.",
        "The earth is hollow and inhabited by an advanced civilization.",
        "Drinking coffee makes you invisible on radar.",
        "Birds are actually surveillance drones deployed by the government.",
        "Eating organic food gives you superpowers instantly.",
        "The government hides free energy technology from the public.",
        "5G networks cause instant mind control in all users.",
        "Ancient aliens built all major world monuments."
    ]
    
    df_real = pd.DataFrame({'text': real_news, 'label': 'REAL'})
    df_fake = pd.DataFrame({'text': fake_news, 'label': 'FAKE'})
    df = pd.concat([df_real, df_fake], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(DATASET_PATH, index=False)
    print("Dataset created")


# ======================= MODEL =======================

def train_model():
    """Train the fake news classification model"""
    global MODEL_CACHE
    
    print("Training model...")
    if not os.path.exists(DATASET_PATH):
        create_synthetic_data()
    
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1, ngram_range=(1, 2))),
        ('clf', PassiveAggressiveClassifier(max_iter=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {score*100:.2f}%')
    
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved")
    MODEL_CACHE = pipeline
    return pipeline

def load_model():
    """Load trained model or train if not exists"""
    global MODEL_CACHE
    
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    
    if os.path.exists(MODEL_PATH):
        MODEL_CACHE = joblib.load(MODEL_PATH)
        return MODEL_CACHE
    else:
        return train_model()

def predict_news(text):
    """Predict if news is REAL or FAKE"""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    model = load_model()
    prediction = model.predict([text])[0]
    
    try:
        decision = model.named_steps['clf'].decision_function(
            model.named_steps['tfidf'].transform([text])
        )
        confidence_score = 1 / (1 + np.exp(-decision[0]))
        classes = model.named_steps['clf'].classes_
        
        if decision[0] > 0:
            predicted_class = classes[-1]
            display_confidence = confidence_score
        else:
            predicted_class = classes[0]
            display_confidence = 1 - confidence_score
        
        display_confidence = max(0.5, min(1.0, display_confidence))
        
    except Exception as e:
        print(f"Error in decision: {e}")
        predicted_class = prediction
        display_confidence = 0.75
        
    return predicted_class, display_confidence

def explain_prediction(text):
    """Returns significant words contributing to the decision"""
    try:
        model = load_model()
        vectorizer = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        
        input_vector = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = input_vector.nonzero()[1]
        coefs = clf.coef_[0]
        
        feature_contributions = []
        for idx in nonzero_indices:
            word = feature_names[idx]
            weight = coefs[idx]
            feature_contributions.append((word, weight))
        
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        result = feature_contributions[:5]
        if not result:
            result = [("no_features", 0.0)]
        return result
    except Exception as e:
        print(f"Error in explain: {e}")
        return [("error", 0.0)]


# ======================= FLASK APP =======================

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        prediction, confidence = predict_news(text)
        explanation = explain_prediction(text)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence*100:.1f}%",
            'explanation': explanation
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


# Initialize model on startup
def init_model():
    """Initialize model on app startup"""
    try:
        if not os.path.exists(DATASET_PATH):
            create_synthetic_data()
        if not os.path.exists(MODEL_PATH):
            train_model()
        else:
            load_model()
    except Exception as e:
        print(f"Warning: Could not initialize model: {e}")

try:
    init_model()
except Exception as e:
    print(f"Init error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
