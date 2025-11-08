from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
import datetime
import time
import cv2
import numpy as np
import csv
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from dataset import create_dataset
from recognition import face_rec
import shutil
from werkzeug.utils import secure_filename
import telepot

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder for fingerprints
UPLOAD_FOLDER = 'static/fingerprints'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to load user data
def load_user_data():
    users = []
    if os.path.exists('Details.csv'):
        with open('Details.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            users = list(reader)

    return users
# Helper function to save user data
def save_user_data(users):
    with open('Details.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Name', 'Phone', 'Email', 'Account_Number', 'IFSC_Code', 'Branch', 'Amount', 'Fingerprint_Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)

@app.route('/')
def index():
    msg = request.args.get('msg')
    return render_template("index.html", msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Get form data
            Id = request.form['Id']
            Name = request.form['Name']
            Phone = request.form['Phone']
            Email = request.form['Email']
            account_number = request.form['account_number']
            ifsc_code = request.form['ifsc_code']
            branch = request.form['branch']
            amount = request.form['amount']
            
            # Handle fingerprint upload
            if 'fingerprint' not in request.files:
                return render_template("register.html", error="No fingerprint file selected")
            
            fingerprint_file = request.files['fingerprint']
            if fingerprint_file.filename == '':
                return render_template("register.html", error="No fingerprint file selected")
            
            # Save fingerprint
            filename = secure_filename(f"{Id}_{Name}_fingerprint.png")
            fingerprint_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            fingerprint_file.save(fingerprint_path)
            
            # Create face dataset (you'll need to implement this)
            create_dataset(Name)
            
            # Save user details
            user_details = {
                'Id': Id,
                'Name': Name,
                'Phone': Phone,
                'Email': Email,
                'Account_Number': account_number,
                'IFSC_Code': ifsc_code,
                'Branch': branch,
                'Amount': amount,
                'Fingerprint_Path': filename
            }
            
            # Save to CSV
            file_exists = os.path.exists('Details.csv')
            with open('Details.csv', 'a', newline='') as csvFile:
                fieldnames = ['Id', 'Name', 'Phone', 'Email', 'Account_Number', 'IFSC_Code', 'Branch', 'Amount', 'Fingerprint_Path']
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(user_details)
            
            # Prepare success message
            msg = [
                'Registration Successful!',
                f'ID: {Id}',
                f'Name: {Name}',
                f'Phone: {Phone}',
                f'Email: {Email}',
                f'Account Number: {account_number}',
                f'Initial Balance: ₹{amount}'
            ]
            
            return redirect(url_for('index', msg=msg))
        
        except Exception as e:
            return render_template("register.html", error=f"An error occurred: {str(e)}")
    
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        Id = request.form['ID']
        Name = request.form['Name']
        otp = request.form['otp']
        if session['otp'] == otp:
            # Perform face recognition (you'll need to implement this)
            recognized_name = face_rec()
            
            if recognized_name == 'Unknown' or recognized_name != Name:
                return render_template("login.html", error="Face recognition failed. Please try again.")
            
            # If face recognition passes, load user data
            user_data = None
            with open('Details.csv', 'r') as csvFile:
                reader = csv.DictReader(csvFile)
                for row in reader:
                    if row['Id'] == Id and row['Name'] == Name:
                        user_data = row
                        break
            
            if not user_data:
                return render_template("login.html", error="User not found")
            
            # Store user in session
            session['user'] = user_data
            
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Entered wrong otp")
    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    
    # Get all accounts for this user
    users = load_user_data()
    accounts = []
    for u in users:
        if u['Id'] == user['Id']:
            accounts.append({
                'account_number': u['Account_Number'],
                'ifsc_code': u['IFSC_Code'],
                'branch': u['Branch'],
                'balance': u['Amount']
            })
    
    return render_template("dashboard.html", user=user, accounts=accounts)

@app.route('/transfer', methods=['POST'])
def transfer():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        user = session['user']
        from_account = request.form['from_account']
        to_account = request.form['to_account']
        amount = float(request.form['amount'])
        
        # Verify fingerprint
        if 'fingerprint' not in request.files:
            return render_template("dashboard.html", 
                                error="No fingerprint file uploaded", 
                                user=user, 
                                accounts=get_user_accounts(user['Id']))
        
        fingerprint_file = request.files['fingerprint']
        if fingerprint_file.filename == '':
            return render_template("dashboard.html", 
                                error="No fingerprint file uploaded", 
                                user=user, 
                                accounts=get_user_accounts(user['Id']))
        
        # Verify fingerprint matches registered fingerprint
        uploaded_filename = secure_filename(fingerprint_file.filename)
        registered_filename = user['Fingerprint_Path']
        
        # Extract fingerprint ID from filenames
        registered_id = registered_filename.split('_')[0]  # Format: ID_Name_fingerprint.png
        uploaded_id = uploaded_filename.split('_')[0]
        
        if registered_id != uploaded_id or not uploaded_filename.endswith('_fingerprint.png'):
            return render_template("dashboard.html", 
                                error="Fingerprint verification failed - mismatch", 
                                user=user, 
                                accounts=get_user_accounts(user['Id']))
        
        # Check balance
        users = load_user_data()
        for u in users:
            if u['Id'] == user['Id'] and u['Account_Number'] == from_account:
                current_balance = float(u['Amount'])
                if amount > current_balance:
                    return render_template("dashboard.html", 
                                        error="Insufficient funds", 
                                        user=user, 
                                        accounts=get_user_accounts(user['Id']))
                
                # Update balance
                u['Amount'] = str(current_balance - amount)
                break
        
        # Save updated data
        save_user_data(users)
        
        # Update session
        session['user'] = next((u for u in users if u['Id'] == user['Id']), user)
        
        return render_template("dashboard.html", 
                            success=f"Successfully transferred ₹{amount} to account {to_account}",
                            user=session['user'],
                            accounts=get_user_accounts(user['Id']))
    
    except Exception as e:
        return render_template("dashboard.html", 
                            error=f"Transfer failed: {str(e)}",
                            user=session.get('user'),
                            accounts=get_user_accounts(session.get('user', {}).get('Id', '')))

def get_user_accounts(user_id):
    users = load_user_data()
    return [{
        'account_number': u['Account_Number'],
        'ifsc_code': u['IFSC_Code'],
        'branch': u['Branch'],
        'balance': u['Amount']
    } for u in users if u['Id'] == user_id]
@app.route('/merchant')
def merchant():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    return render_template("merchant.html", user=user, accounts=get_user_accounts(user['Id']))
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/get_data')
def get_data():
    import random
    otp = str(random.randint(1111, 9999))
    bot = telepot.Bot('7370932535:AAGU8MFvW8SU0U-_Ai0LgLQzDRTKcwCv23Y')
    bot.sendMessage('5145717462', str(otp))
    session['otp'] = otp
    print(otp)
    return jsonify('OTP sent')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
