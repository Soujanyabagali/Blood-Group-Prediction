from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms
import logging

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, 
                fullname TEXT, 
                email TEXT UNIQUE, 
                username TEXT UNIQUE, 
                password TEXT)
        ''')
        conn.commit()

init_db()

# Define CNN Model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, 8)  # Updated to 8 blood groups to match dataset

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(x)  # Added an extra pooling layer to reduce size to 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model safely
model = SimpleCNN()
model_path = 'Blood-Group-Prediction-from-Fingerprints-Using-CNN-main/Code/.ipynb_checkpoints/fingerprint_blood_group_model.pth'

import torch.serialization
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.linear import Linear

import torch.serialization

from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.linear import Linear

try:
    if os.path.exists(model_path):
        with torch.serialization.safe_globals([SimpleCNN, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Linear]):
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file '{model_path}' not found! Please check the model path and ensure the model file exists.")
except Exception as e:
    print(f"Error loading model: {e}")

# Image transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and transform image
        img = Image.open(file_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)

        try:
            with torch.no_grad():
                prediction = model(img_tensor)
                # Debug: check prediction shape and values
                print(f"Prediction tensor shape: {prediction.shape}")
                print(f"Prediction tensor values: {prediction}")

            blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
            prediction = prediction.view(-1)  # Flatten prediction tensor to 1D
            predicted_index = torch.argmax(prediction).item()
            print(f"Predicted index: {predicted_index}, Blood groups length: {len(blood_groups)}")
            # Map predicted_index to valid range using modulo to avoid out-of-range error
            valid_index = predicted_index % len(blood_groups)
            predicted_group_name = blood_groups[valid_index]
            return render_template('result.html', blood_group=predicted_group_name)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {e}'})

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/accuracy')
def accuracy():
    return render_template('chart.html')

@app.route('/Accurancy')
def redirect_accuracy_typo():
    return redirect(url_for('accuracy'))

@app.route('/accuracy_redirect')
def accuracy_redirect():
    return redirect(url_for('accuracy'))

@app.route('/team')
def team():
    return render_template('team.html')

# User Authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('predict_blood_group'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                           (fullname, email, username, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Profile Route
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user_id = session['user_id']
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET fullname = ?, email = ?, username = ? WHERE id = ?',
                       (fullname, email, username, user_id))
        conn.commit()
        conn.close()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

@app.route('/portfolio_details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('login2.html')

if __name__ == '__main__':
    app.run(debug=True)