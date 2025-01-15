# Importing necessary modules
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, make_response
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
import mysql.connector
from mysql.connector import pooling
from openai import OpenAI
import logging
from logging.handlers import RotatingFileHandler
import openai
import random
import traceback
import string
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
# Setup logging
logging.basicConfig(
    filename='user_activity.log',  # Log file for user activities
    level=logging.INFO,  # Set log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format with timestamp
)
# Set OpenAI API key from environment variable
openapikey = os.getenv("OPENAI_API_KEY")
# Init
client = OpenAI(api_key=openapikey)


# Initialize Flask app
app = Flask(__name__)

# Set Flask secret key (ensure consistency with your .env)
app.secret_key = os.getenv('APP_SECRET_KEY')

# Configure Flask-Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))  # Convert to integer
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'  # Convert string to boolean
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL') == 'True'  # Convert string to boolean
mail = Mail(app)

# Configure Flask-SQLAlchemy for MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS') == 'False'  # Convert string to boolean
db = SQLAlchemy(app)

# User model
class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    user_name = db.Column(db.String(120), nullable=False)  # Added user_name field
    otp = db.Column(db.String(6), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

class DBHandler(logging.Handler):
    def __init__(self, db):
        logging.Handler.__init__(self)
        self.db = db

    def emit(self, record):
        log_entry = self.format(record)
        try:
            # Add log entry to the database
            user_id = getattr(record, 'user_id', None)
            user_name = getattr(record, 'user_name', None)
            activity = record.getMessage()  # Get the log message
            log_level = record.levelname
            timestamp = datetime.now()

            # Insert log into the activity_logs table
            query = "INSERT INTO activity_logs (user_id, user_name, activity, log_level, timestamp) VALUES (%s, %s, %s, %s, %s)"
            values = (user_id, user_name, activity, log_level, timestamp)
            connection = connect_db()  # Get DB connection from your connection pool
            cursor = connection.cursor()
            cursor.execute(query, values)
            connection.commit()
            cursor.close()
            connection.close()

        except Exception as e:
            logging.error(f"Failed to log activity to the database: {e}")

# Initialize your logging handler to store logs in the DB
db_handler = DBHandler(db)

# Set logging level and format
db_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
db_handler.setFormatter(formatter)

# Add handler to the Flask app logger
app.logger.addHandler(db_handler)


# Generate a 6-digit OTP
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

connection_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=20,  # Number of connections in the pool
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
    # port=1199
)

# Connecting MySQL database
def connect_db():
    try:
        # Get a connection from the pool
        connection = connection_pool.get_connection()
        
        # Check if the connection is established
        if connection.is_connected():
            return connection
        else:
            logging.error("Error: Unable to obtain a connection from the pool.")
            return None

    except mysql.connector.Error as err:
        logging.error(f"Database connection error: {err}")
        return None

# Mapping the questions table and the responses tables based on the question_id
def get_questions_by_type(q_type):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT id, question FROM questions WHERE q_type = %s"
    cursor.execute(query, (q_type,))
    questions = cursor.fetchall()
    cursor.close()
    connection.close()
    return questions

# Helper function to filter data based on date range
def filter_by_date_range(query, start_date=None, end_date=None):
    if start_date:
        query += f" AND r.created_at >= '{start_date}'"
    if end_date:
        query += f" AND r.created_at <= '{end_date}'"
    return query

# Helper function to filter website data based on data range
def filter_csat_by_date_range(query, start_date=None, end_date=None):
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
    elif start_date:
        query += " AND date >= %s"
    elif end_date:
        query += " AND date <= %s"
    return query

def filter_ces_by_date_range(query, start_date=None, end_date=None):
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
    elif start_date:
        query += " AND date >= %s"
    elif end_date:
        query += " AND date <= %s"
    return query

# Function to get CSAT data with date filtering
def get_csat_data(question_id, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    WHERE r.question_id = %s
    """
    
    query = filter_by_date_range(query, start_date, end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, (question_id,))
    csat_df = cursor.fetchall()
    
    if not csat_df:
        print("No CSAT data found for the specified date range.")  # Debug statement
    
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_csat = len(csat_df)

    for row in csat_df:
        rating = row['rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_csat

# Function to get NPS data with date filtering
def get_nps_data(question_id, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    WHERE r.question_id = %s
    """
    
    query = filter_by_date_range(query, start_date, end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, (question_id,))
    nps_df = cursor.fetchall()
    
    if not nps_df:
        print("No NPS data found for the specified date range.")  # Debug statement
    
    cursor.close()
    connection.close()

    detractors = 0
    passive = 0
    promoters = 0
    total_nps = len(nps_df)

    for row in nps_df:
        rating = row['rating']
        if rating is not None:
            if rating <= 6:
                detractors += 1
            elif rating in [7, 8]:
                passive += 1
            elif rating >= 9:
                promoters += 1

    detractors_percentage = (detractors / total_nps) * 100 if total_nps > 0 else 0
    passives_percentage = (passive / total_nps) * 100 if total_nps > 0 else 0
    promoters_percentage = (promoters / total_nps) * 100 if total_nps > 0 else 0
    nps_percentage = promoters_percentage - detractors_percentage

    return total_nps, detractors, passive, promoters, detractors_percentage, passives_percentage, promoters_percentage, nps_percentage

# Function to get overall CSAT data
def get_overall_csat_data(question_id):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    WHERE r.question_id = %s
    """
    
    cursor.execute(query, (question_id,))
    csat_df = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_csat = len(csat_df)

    for row in csat_df:
        rating = row['rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_csat

#get overall website csat data
def get_overall_website_csat_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Website_Duplicates
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    csat_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(csat_data)

    for row in csat_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses

#get website csat data by location
def get_website_csat_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Website_Duplicates
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_csat_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses

# Function to get overall NPS data
def get_overall_nps_data(question_id):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    WHERE r.question_id = %s
    """
    
    cursor.execute(query, (question_id,))
    nps_df = cursor.fetchall()
    cursor.close()
    connection.close()

    detractors = 0
    passive = 0
    promoters = 0
    total_nps = len(nps_df)

    for row in nps_df:
        rating = row['rating']
        if rating is not None:
            if rating <= 6:
                detractors += 1
            elif rating in [7, 8]:
                passive += 1
            elif rating >= 9:
                promoters += 1

    detractors_percentage = (detractors / total_nps) * 100 if total_nps > 0 else 0
    passives_percentage = (passive / total_nps) * 100 if total_nps > 0 else 0
    promoters_percentage = (promoters / total_nps) * 100 if total_nps > 0 else 0
    nps_percentage = promoters_percentage - detractors_percentage

    return total_nps, detractors, passive, promoters, detractors_percentage, passives_percentage, promoters_percentage, nps_percentage

#get overall website ces data
def get_overall_ces_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Website_Duplicates
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing CES query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    ces_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(ces_data)

    for row in ces_data:
        rating = row['ces_rating']
        if rating is not None and 1 <= rating <= 5:  # Ensure the rating is valid
            rating_counts[rating] += 1
            if rating >= 4:
                positive_responses += 1

    return rating_counts, positive_responses, total_responses

#get website ces data by location
# Function to get CES data by location
def get_website_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Website_Duplicates
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_ces_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing CES query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['ces_rating']
        if rating is not None and 1 <= rating <= 5:  # Ensure the rating is valid
            rating_counts[rating] += 1
            if rating >= 4:
                positive_responses += 1

    return rating_counts, positive_responses, total_responses

def classify_text_openai(text, timeout_duration=15):
    try:
        # Fetch API key from environment variable
        # client = os.getenv("OPENAI_API_KEY")
        if not client:
            raise ValueError("OpenAI API key is not set. Please set it in the environment variables.")
        
        prompt = f"Classify the following text into one of these categories: Positive, Negative, or Suggestion.\n\nText: \"{text}\"\n\nCategory:"
        # Proceed with the OpenAI API request using the ChatCompletion endpoint
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Correct model name
            messages=[
                {"role": "system", "content": "You are a text classification assistant."},
                # {"role": "user", "content": "Classify the following text into one of these categories: Positive, Negative, or Suggestion.\n\nText: \"{text}\"\n\nCategory:"}
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Increase token limit to handle larger comments
            temperature=0,
            timeout=5  # Set a timeout for the API call
        )
        # Extract and return the classified category

        # response_dict = response.to_dict()
        return  response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error in CSAT classifications endpoint: {repr(e)}")
        return jsonify({"error": f"An error occurred: {repr(e),traceback.format_exc()}"})

#Api to get CSAT details based on Loaction
def get_csat_data_by_location(location, question_id, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    JOIN loan_applications la ON r.lead_id = la.lan
    WHERE r.question_id = %s AND la.branch = %s
    """
    
    # Apply date range filtering if provided
    query = filter_by_date_range(query, start_date, end_date)
    
    params = (question_id, location)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, params)
    csat_df = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_csat = len(csat_df)

    for row in csat_df:
        rating = row['rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_csat


#Getting NPS Data by location
def get_nps_data_by_location(location, question_id, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = f"""
    SELECT r.rating
    FROM responses r
    JOIN loan_applications la ON r.lead_id = la.lan
    WHERE r.question_id = %s AND la.branch = %s
    """
    
    # Apply date range filtering if provided
    query = filter_by_date_range(query, start_date, end_date)
    
    params = (question_id, location)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, params)
    nps_df = cursor.fetchall()
    cursor.close()
    connection.close()

    detractors = 0
    passive = 0
    promoters = 0
    total_nps = len(nps_df)

    for row in nps_df:
        rating = row['rating']
        if rating is not None:
            if rating <= 6:
                detractors += 1
            elif rating in [7, 8]:
                passive += 1
            elif rating >= 9:
                promoters += 1

    detractors_percentage = (detractors / total_nps) * 100 if total_nps > 0 else 0
    passives_percentage = (passive / total_nps) * 100 if total_nps > 0 else 0
    promoters_percentage = (promoters / total_nps) * 100 if total_nps > 0 else 0
    nps_percentage = promoters_percentage - detractors_percentage

    return total_nps, detractors, passive, promoters, detractors_percentage, passives_percentage, promoters_percentage, nps_percentage

#Classifying the comments in the 3categories and fetching the comments,name, email,mobile, branch and date(comment)
def classify_suggestions_for_survey(q_type):
    try:
        connection = connect_db()
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT DISTINCT r.text_response, r.lead_id, r.name, r.email, r.mobile, la.branch, r.created_at
        FROM responses r
        JOIN loan_applications la ON r.lead_id = la.lan
        JOIN questions q ON r.question_id = q.id
        WHERE q.q_type = %s AND r.text_response IS NOT NULL 
              AND r.text_response != 'NULL' AND r.text_response != 'None'
        """
        cursor.execute(query, (q_type,))
        suggestions_df = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    unique_responses = set()
    classifications = {
        'Positive': [],
        'Negative': [],
        'Suggestion': []
    }

    for row in suggestions_df:
        text = row['text_response'].strip()
        lead_id = row['lead_id']
        name = row['name']
        email = row['email']
        mobile = row['mobile']
        branch = row['branch']
        created_at = row['created_at']
        
        if len(text.split()) > 2:
            unique_key = (text, lead_id)
            if unique_key not in unique_responses:
                unique_responses.add(unique_key)
                formatted_date = created_at.strftime('%Y-%m-%d')
                category = classify_text_openai(text)
                comment = {
                    'text_response': text,
                    'lead_id': lead_id,
                    'name': name,
                    'email': email,
                    'mobile': mobile,
                    'branch': branch,
                    'date': formatted_date
                }
                if category in classifications:
                    classifications[category].append(comment)

    counts = {
        'positive': len(classifications['Positive']),
        'negative': len(classifications['Negative']),
        'suggestions': len(classifications['Suggestion'])
    }

    return {
        'positive_comments': classifications['Positive'],
        'negative_comments': classifications['Negative'],
        'suggestions': classifications['Suggestion'],
        'counts': counts
    }

##Classification for Login comments(CSAT)##
# Classifying the feedback and fetching the comments, name, email, mobile, city, and date
def classify_suggestions_for_csat_login():
    try:
        connection = connect_db()  # Assuming connect_db() is your DB connection method
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT DISTINCT l.csat_response, l.lan, l.user_id, l.email, l.mobile, l.city, l.created_at
        FROM Login l
        WHERE l.csat_response IS NOT NULL 
              AND l.csat_response != 'NULL' AND l.csat_response != 'None'
        """
        cursor.execute(query)
        suggestions_df = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    unique_responses = set()
    classifications = {
        'Positive': [],
        'Negative': [],
        'Suggestion': []
    }

    for row in suggestions_df:
        text = row['csat_response'].strip()
        lan = row['lan']
        user_id = row['user_id']
        email = row['email']
        mobile = row['mobile']
        city = row['city']
        created_at = row['created_at']
        
        # Check if text has more than 2 words
        if len(text.split()) > 2:
            unique_key = (text, lan)
            if unique_key not in unique_responses:
                unique_responses.add(unique_key)
                formatted_date = created_at.strftime('%Y-%m-%d')
                
                # Call the classification function
                category = classify_text_openai(text)
                
                if category:  # Only proceed if classification is successful
                    comment = {
                        'csat_response': text,
                        'lan': lan,
                        'user_id': user_id,
                        'email': email,
                        'mobile': mobile,
                        'city': city,
                        'date': formatted_date
                    }

                    if category in classifications:
                        classifications[category].append(comment)

    counts = {
        'positive': len(classifications['Positive']),
        'negative': len(classifications['Negative']),
        'suggestions': len(classifications['Suggestion'])
    }

    return {
        'positive_comments': classifications['Positive'],
        'negative_comments': classifications['Negative'],
        'suggestions': classifications['Suggestion'],
        'counts': counts
    }

#get overall login ces data
def get_overall_login_ces_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Login
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    ces_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(ces_data)

    for row in ces_data:
        rating = row['ces_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses

#get login ces data by location
def get_login_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Login
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_ces_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['ces_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses


##Classification for Login comments(CES)##
# Classifying the feedback and fetching the comments, name, email, mobile, city, and date
def classify_suggestions_for_login():
    try:
        connection = connect_db()  # Assuming connect_db() is your DB connection method
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT DISTINCT l.feedback, l.lan, l.user_id, l.email, l.mobile, l.city, l.created_at
        FROM Login l
        WHERE l.feedback IS NOT NULL 
              AND l.feedback != 'NULL' AND l.feedback != 'None'
        """
        cursor.execute(query)
        suggestions_df = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    unique_responses = set()
    classifications = {
        'Positive': [],
        'Negative': [],
        'Suggestion': []
    }

    for row in suggestions_df:
        text = row['feedback'].strip()
        lan = row['lan']
        user_id = row['user_id']
        email = row['email']
        mobile = row['mobile']
        city = row['city']
        created_at = row['created_at']
        
        # Check if text has more than 2 words
        if len(text.split()) > 2:
            unique_key = (text, lan)
            if unique_key not in unique_responses:
                unique_responses.add(unique_key)
                formatted_date = created_at.strftime('%Y-%m-%d')
                
                # Call the classification function
                category = classify_text_openai(text)
                
                if category:  # Only proceed if classification is successful
                    comment = {
                        'feedback': text,
                        'lan': lan,
                        'user_id': user_id,
                        'email': email,
                        'mobile': mobile,
                        'city': city,
                        'date': formatted_date
                    }

                    if category in classifications:
                        classifications[category].append(comment)

    counts = {
        'positive': len(classifications['Positive']),
        'negative': len(classifications['Negative']),
        'suggestions': len(classifications['Suggestion'])
    }

    return {
        'positive_comments': classifications['Positive'],
        'negative_comments': classifications['Negative'],
        'suggestions': classifications['Suggestion'],
        'counts': counts
    }

#get overall website csat data
def get_overall_login_ces_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Login
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    csat_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(csat_data)

    for row in csat_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses


#get website csat data by location
def get_login_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Login
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_csat_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses

#Getting the number of counts for positive_comments, negative_comments and suggestions
def get_comments_count_by_branch(q_type):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT la.branch, COUNT(r.text_response) as comments_count
    FROM responses r
    JOIN loan_applications la ON r.lead_id = la.lan
    JOIN questions q ON r.question_id = q.id
    WHERE q.q_type = %s AND r.text_response IS NOT NULL 
          AND r.text_response != 'NULL' AND r.text_response != 'None'
    GROUP BY la.branch
    """
    
    cursor.execute(query, (q_type,))
    branch_comments = cursor.fetchall()
    cursor.close()
    connection.close()

    return branch_comments

def filter_ces_by_date_range(query, start_date=None, end_date=None):
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
    elif start_date:
        query += " AND date >= %s"
    elif end_date:
        query += " AND date <= %s"
    return query

def get_overall_ces_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Website_Duplicates
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing CES query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    ces_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(ces_data)

    for row in ces_data:
        rating = row['ces_rating']
        if rating is not None and 1 <= rating <= 5:  # Ensure the rating is valid
            rating_counts[rating] += 1
            if rating >= 4:
                positive_responses += 1

    return rating_counts, positive_responses, total_responses

#get website ces data by location
# Function to get CES data by location
def get_website_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT ces_rating
    FROM Website_Duplicates
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_ces_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing CES query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['ces_rating']
        if rating is not None and 1 <= rating <= 5:  # Ensure the rating is valid
            rating_counts[rating] += 1
            if rating >= 4:
                positive_responses += 1

    return rating_counts, positive_responses, total_responses

def filter_csat_by_date_range(query, start_date=None, end_date=None):
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
    elif start_date:
        query += " AND date >= %s"
    elif end_date:
        query += " AND date <= %s"
    return query

#get overall website csat data
def get_overall_website_csat_data(start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Website_Duplicates
    WHERE 1=1
    """
    
    params = []
    
    # Apply date range filtering if provided
    if start_date and end_date:
        query += " AND date BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        query += " AND date >= %s"
        params.append(start_date)
    elif end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    csat_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(csat_data)

    for row in csat_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses


#get website csat data by location
def get_website_data_by_location(location, start_date=None, end_date=None):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT csat_rating
    FROM Website_Duplicates
    WHERE city = %s
    """
    
    # Apply date range filtering if provided
    query = filter_csat_by_date_range(query, start_date, end_date)
    
    params = [location]
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    
    print("Executing query:", query)  # Debug statement
    cursor.execute(query, tuple(params))
    website_data = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_responses = len(website_data)

    for row in website_data:
        rating = row['csat_rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    return rating_counts, positive_responses, total_responses


def get_ces_comments_count_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    query = """
    SELECT l.city, COUNT(l.csat_response) as comments_count
    FROM Login l
    WHERE l.csat_response IS NOT NULL 
          AND l.csat_response != 'NULL' 
          AND l.csat_response != 'None'
    GROUP BY l.city
    """
    
    cursor.execute(query)
    city_comments = cursor.fetchall()
    cursor.close()
    connection.close()

    return city_comments


def get_comments_count_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT l.city, COUNT(l.feedback) as comments_count
    FROM Login l
    WHERE l.feedback IS NOT NULL 
          AND l.feedback != 'NULL' 
          AND l.feedback != 'None'
    GROUP BY l.city
    """
    
    cursor.execute(query)
    city_comments = cursor.fetchall()
    cursor.close()
    connection.close()

    return city_comments

#API endpoint for getting website csat data by location
@app.route('/api/website/csat/location', methods=['GET'])
def website_csat_location_api():
    city = request.args.get('city')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not city:
        return jsonify({'error': 'city is required'}), 400

    rating_counts, positive_responses, total_responses = get_website_data_by_location(
        city, start_date, end_date
    )

    csat_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'csat_rating_counts': rating_counts,
        'csat_positive_responses': positive_responses,
        'csat_percentage': csat_percentage,
        'total_responses': total_responses
    })

@app.route('/api/website/csatcity', methods=['GET'])
def get_csat_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT city, COUNT(*) as comments_count
    FROM Website_Duplicates
    GROUP BY city
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        results = []
    finally:
        cursor.close()
        connection.close()
    
    return jsonify(results)


#New API for CSAT details over location(date filter included)
@app.route('/api/survey/csat/location', methods=['GET'])
def csat_location_api():
    question_id = request.args.get('question_id')
    branch = request.args.get('branch')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not question_id or not branch:
        return jsonify({'error': 'question_id and branch are required'}), 400

    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    # Base query to get CSAT data for a specific location
    query = f"""
    SELECT r.rating
    FROM responses r
    JOIN loan_applications la ON r.lead_id = la.lan
    WHERE r.question_id = %s AND la.branch = %s
    """
    
    # Apply date filtering if provided
    query = filter_by_date_range(query, start_date, end_date)
    
    cursor.execute(query, (question_id, branch))
    csat_df = cursor.fetchall()
    cursor.close()
    connection.close()

    rating_counts = {i: 0 for i in range(1, 6)}
    positive_responses = 0
    total_csat = len(csat_df)

    for row in csat_df:
        rating = row['rating']
        rating_counts[rating] += 1
        if rating >= 4:
            positive_responses += 1

    csat_percentage = (positive_responses / total_csat) * 100 if total_csat > 0 else 0
    
    return jsonify({
        'csat_rating_counts': rating_counts,
        'csat_positive_responses': positive_responses,
        'csat_percentage': csat_percentage,
        'total_csat': total_csat
    })

@app.route('/api/website/ces/location', methods=['GET'])
def website_ces_location_api():
    city = request.args.get('city')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not city:
        return jsonify({'error': 'city is required'}), 400

    rating_counts, positive_responses, total_responses = get_website_data_by_location(
        city, start_date, end_date
    )

    ces_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'ces_rating_counts': rating_counts,
        'ces_positive_responses': positive_responses,
        'ces_percentage': ces_percentage,
        'total_responses': total_responses
    })

# API endpoint for getting overall CES data
@app.route('/api/website/ces/overall', methods=['GET'])
def website_ces_overall_api():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    rating_counts, positive_responses, total_responses = get_overall_ces_data(
        start_date, end_date
    )

    ces_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'ces_rating_counts': rating_counts,
        'ces_positive_responses': positive_responses,
        'ces_percentage': ces_percentage,
        'total_responses': total_responses
    })


@app.route('/api/website/cescity', methods=['GET'])
def get_ces_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT city, COUNT(*) as comments_count
    FROM Website_Duplicates
    GROUP BY city
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        results = []
    finally:
        cursor.close()
        connection.close()
    
    return jsonify(results)
#Get NPS data by location
#Endpoint for getting NPS data by location
@app.route('/api/survey/nps/location', methods=['GET'])
def nps_location_api():
    question_id = request.args.get('question_id')
    branch = request.args.get('branch')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not question_id or not branch:
        return jsonify({'error': 'question_id and branch are required'}), 400

    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    # Base query to get NPS data for a specific location
    query = f"""
    SELECT r.rating
    FROM responses r
    JOIN loan_applications la ON r.lead_id = la.lan
    WHERE r.question_id = %s AND la.branch = %s
    """
    
    # Apply date filtering if provided
    query = filter_by_date_range(query, start_date, end_date)
    
    cursor.execute(query, (question_id, branch))
    nps_df = cursor.fetchall()
    cursor.close()
    connection.close()

    detractors = 0
    passive = 0
    promoters = 0
    total_nps = len(nps_df)

    for row in nps_df:
        rating = row['rating']
        if rating is not None:
            if rating <= 6:
                detractors += 1
            elif rating in [7, 8]:
                passive += 1
            elif rating >= 9:
                promoters += 1

    detractors_percentage = (detractors / total_nps) * 100 if total_nps > 0 else 0
    passives_percentage = (passive / total_nps) * 100 if total_nps > 0 else 0
    promoters_percentage = (promoters / total_nps) * 100 if total_nps > 0 else 0
    nps_percentage = promoters_percentage - detractors_percentage
    
    return jsonify({
        'detractors': detractors,
        'passive': passive,
        'promoters': promoters,
        'detractors_percentage': detractors_percentage,
        'passives_percentage': passives_percentage,
        'promoters_percentage': promoters_percentage,
        'nps_percentage': nps_percentage,
        'total_nps': total_nps
    })


# API to get CSAT data with date filtering
@app.route('/api/survey/csat', methods=['GET'])
def csat_api():
    question_id = request.args.get('question_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    print("Received start date:", start_date)  # Debug statement
    print("Received end date:", end_date)  # Debug statement
    
    csat_rating_counts, csat_positive_responses, total_csat = get_csat_data(question_id, start_date, end_date)
    csat_percentage = (csat_positive_responses / total_csat) * 100 if total_csat > 0 else 0
    
    return jsonify({
        'csat_rating_counts': csat_rating_counts,
        'csat_positive_responses': csat_positive_responses,
        'csat_percentage': csat_percentage,
        'total_csat': total_csat
    })

# API to get NPS data with date filtering
@app.route('/api/survey/nps', methods=['GET'])
def nps_api():
    question_id = request.args.get('question_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    print("Received start date:", start_date)  # Debug statement
    print("Received end date:", end_date)  # Debug statement
    
    total_nps, detractors, passive, promoters, detractors_percentage, passives_percentage, promoters_percentage, nps_percentage = get_nps_data(question_id, start_date, end_date)
    
    return jsonify({
        'total_nps': total_nps,
        'detractors': detractors,
        'passive': passive,
        'promoters': promoters,
        'detractors_percentage': detractors_percentage,
        'passives_percentage': passives_percentage,
        'promoters_percentage': promoters_percentage,
        'nps_percentage': nps_percentage
    })

# New API for overall CSAT data
@app.route('/api/survey/overall_csat', methods=['GET'])
def overall_csat_api():
    question_id = request.args.get('question_id')
    
    csat_rating_counts, csat_positive_responses, total_csat = get_overall_csat_data(question_id)
    csat_percentage = (csat_positive_responses / total_csat) * 100 if total_csat > 0 else 0
    
    return jsonify({
        'csat_rating_counts': csat_rating_counts,
        'csat_positive_responses': csat_positive_responses,
        'csat_percentage': csat_percentage,
        'total_csat': total_csat
    })

# New API for overall NPS data
@app.route('/api/survey/overall_nps', methods=['GET'])
def overall_nps_api():
    question_id = request.args.get('question_id')
    
    total_nps, detractors, passive, promoters, detractors_percentage, passives_percentage, promoters_percentage, nps_percentage = get_overall_nps_data(question_id)
    
    return jsonify({
        'total_nps': total_nps,
        'detractors': detractors,
        'passive': passive,
        'promoters': promoters,
        'detractors_percentage': detractors_percentage,
        'passives_percentage': passives_percentage,
        'promoters_percentage': promoters_percentage,
        'nps_percentage': nps_percentage
    })

#API endpoint for csat responses classification
@app.route('/api/survey/csat/classifications', methods=['GET'])
def csat_classifications():
    try:
        result = classify_suggestions_for_survey('CSAT')
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in CSAT classifications endpoint: {repr(e)}")
        return jsonify({"error": f"An error occurred: {repr(e),traceback.format_exc()}"})


#API endpoint for nps responses classification
@app.route('/api/survey/nps/classifications', methods=['GET'])
def nps_classifications():
    try:
        result = classify_suggestions_for_survey('NPS')
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in CSAT classifications endpoint: {repr(e)}")
        return jsonify({"error": f"An error occurred: {repr(e),traceback.format_exc()}"})

#API endpoint for data based on question_id
@app.route('/api/survey/questions', methods=['GET'])
def questions_api():
    q_type = request.args.get('q_type')
    if not q_type:
        return jsonify({'error': 'q_type is required'}), 400
    
    questions = get_questions_by_type(q_type)
    return jsonify(questions)

#API endpoint for csat branch-wise details
@app.route('/api/survey/csatbranch', methods=['GET'])
def csat_branch_api():
    branch_comments = get_comments_count_by_branch('CSAT')
    return jsonify(branch_comments)

#API endpoint for nps branch-wise details
@app.route('/api/survey/npsbranch', methods=['GET'])
def nps_branch_api():
    branch_comments = get_comments_count_by_branch('NPS')
    return jsonify(branch_comments)

@app.route('/api/login/ces/location', methods=['GET'])
def login_ces_location_api():
    city = request.args.get('city')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not city:
        return jsonify({'error': 'city is required'}), 400

    rating_counts, positive_responses, total_responses = get_login_data_by_location(
        city, start_date, end_date
    )

    ces_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'ces_rating_counts': rating_counts,
        'ces_positive_responses': positive_responses,
        'ces_percentage': ces_percentage,
        'total_responses': total_responses
    })

#API endpoint for getting overall ces data
@app.route('/api/login/ces/overall', methods=['GET'])
def login_ces_overall_api():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    rating_counts, positive_responses, total_responses = get_overall_login_ces_data(
        start_date, end_date
    )

    ces_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'ces_rating_counts': rating_counts,
        'ces_positive_responses': positive_responses,
        'ces_percentage': ces_percentage,
        'total_responses': total_responses
    })

@app.route('/api/login/cescity', methods=['GET'])
def get_loginces_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT city, COUNT(*) as comments_count
    FROM Login
    GROUP BY city
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        results = []
    finally:
        cursor.close()
        connection.close()
    
    return jsonify(results)

@app.route('/api/login/ces/classifications', methods=['GET'])
def login_ces_classifications():
    try:
        result = classify_suggestions_for_csat_login()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in Login CSAT classifications endpoint: {repr(e)}")
        return jsonify({"error": f"An error occurred: {repr(e),traceback.format_exc()}"})


#API endpoint for getting website csat data by location
@app.route('/api/login/csat/location', methods=['GET'])
def login_csat_location_api():
    city = request.args.get('city')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Validate inputs
    if not city:
        return jsonify({'error': 'city is required'}), 400

    rating_counts, positive_responses, total_responses = get_login_data_by_location(
        city, start_date, end_date
    )

    csat_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'csat_rating_counts': rating_counts,
        'csat_positive_responses': positive_responses,
        'csat_percentage': csat_percentage,
        'total_responses': total_responses
    })

#API endpoint for getting overall csat data
@app.route('/api/login/csat/overall', methods=['GET'])
def login_csat_overall_api():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    rating_counts, positive_responses, total_responses = get_overall_login_ces_data(
        start_date, end_date
    )

    csat_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'csat_rating_counts': rating_counts,
        'csat_positive_responses': positive_responses,
        'csat_percentage': csat_percentage,
        'total_responses': total_responses
    })

@app.route('/api/login/csatcity', methods=['GET'])
def get_logincsat_by_city():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT city, COUNT(*) as comments_count
    FROM Login
    GROUP BY city
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        results = []
    finally:
        cursor.close()
        connection.close()
    
    return jsonify(results)

@app.route('/api/login/csat/classifications', methods=['GET'])
def login_csat_classifications():
    try:
        result = classify_suggestions_for_login()
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in Login CES classifications endpoint: {repr(e)}")
        return jsonify({"error": f"An error occurred: {repr(e),traceback.format_exc()}"})

##to get login csat branch counts
@app.route('/api/login/csatbranch', methods=['GET'])
def login_csat_branch_api():
    city_comments = get_comments_count_by_city()
    return jsonify(city_comments)

##to get login ces branch counts
@app.route('/api/login/cesbranch', methods=['GET'])
def ces_branch_api():
    city_comments = get_ces_comments_count_by_city()
    return jsonify(city_comments)

# API endpoint to send OTP to the defined users
@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Check if the email exists in the database
    user = Users.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'Email not found'}), 404

    otp = generate_otp()

    # Update the OTP in the database
    user.otp = otp
    db.session.commit()

    # Send the OTP email
    try:
        msg = Message('Your OTP Code', sender=os.getenv('MAIL_USERNAME'), recipients=[email])
        msg.body = f'Your OTP code is {otp}.'
        mail.send(msg)
    except Exception as e:
        return jsonify({'error': 'Failed to send OTP. Please try again later.'}), 500

    # Store email in cookies with expiration
    resp = make_response(jsonify({'message': 'OTP sent successfully'}))
    resp.set_cookie('email', email, max_age=600)  # Cookie expires in 10 minutes
    return resp

#API endpoint to check weather a valid otp is entered or not
# API endpoint to check whether a valid OTP is entered or not
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get('email')
    otp = data.get('otp')

    if not email or not otp:
        return jsonify({'error': 'Email and OTP are required'}), 400

    # Check if the email exists
    user = Users.query.filter_by(email=email).first()
    if not user:
        return jsonify({'valid': False}), 400

    # Ensure both OTPs are stripped of extra spaces
    user_otp = user.otp.strip() if user.otp else ''
    otp = otp.strip()

    if user_otp != otp:
        return jsonify({'valid': False}), 400

    # Clear the OTP after successful verification
    user.otp = None
    db.session.commit()

    # Set logged_in cookie
    resp = make_response(jsonify({'valid': True}))
    resp.set_cookie('logged_in', 'true', max_age=1800)  # Cookie expires in 30 minutes

    # Set user_name cookie
    resp.set_cookie('user_name', user.user_name, max_age=1800)  # Cookie expires in 30 minutes

    return resp

#API endpoint for getting overall csat data
@app.route('/api/website/csat/overall', methods=['GET'])
def website_csat_overall_api():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    rating_counts, positive_responses, total_responses = get_overall_website_csat_data(
        start_date, end_date
    )

    csat_percentage = (positive_responses / total_responses) * 100 if total_responses > 0 else 0
    
    return jsonify({
        'csat_rating_counts': rating_counts,
        'csat_positive_responses': positive_responses,
        'csat_percentage': csat_percentage,
        'total_responses': total_responses
    })

#The redirection part when directed to app.py
@app.route('/')
def root():
    return redirect(url_for('login'))

# Login page
@app.route('/cxm/')
def login():
    return render_template('login.html')

# OTP page
@app.route('/cxm/otp')
def otp_page():
    # Check if the user is already logged in
    if request.cookies.get('logged_in') == 'true':
        return redirect(url_for('source_name'))
    return render_template('otp.html')

# Source name selection page
@app.route('/cxm/source_name')
def source_name():
    # Check if the user is logged in
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the source name page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        return render_template('source_name.html', user_name=user_name)
    return redirect(url_for('login'))

# Customer portal page
@app.route('/cxm/customer_portal')
def customer_portal():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the customer portal."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('customer_portal.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Website page
@app.route('/cxm/website')
def website():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the website page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('website.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Customer Portal -> CSAT page
@app.route('/cxm/csat')
def csat():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the CSAT page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('csat.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Website -> CSAT page
@app.route('/cxm/website_csat')
def website_csat():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the website CSAT page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('website_csat.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Website -> CES page
@app.route('/cxm/website_ces')
def website_ces():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the website CES page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('website_ces.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Customer Portal -> NPS page
@app.route('/cxm/nps')
def nps():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the NPS page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('nps.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))


# SourceName -> Login(when user selects Login as SourceName)
@app.route('/cxm/Login')
def lns_login():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the website page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('lns.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Login -> CSAT page
@app.route('/cxm/login_csat')
def login_csat():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the Csat(Login) page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('login_csat.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Login -> CES Page
@app.route('/cxm/login_ces')
def login_ces():
    if request.cookies.get('logged_in') == 'true':
        user_id = request.cookies.get('user_id')
        user_name = request.cookies.get('user_name')

        # Log user activity
        log_message = f"User {user_name} (ID: {user_id}) accessed the CES(Login) page."
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

        api_key = os.getenv('API_URL')
        return render_template('login_ces.html', user_name=user_name, api_key=api_key)
    return redirect(url_for('login'))

# Logout page
@app.route('/cxm/logout')
def logout():
    if 'user_name' in request.cookies:
        user_name = request.cookies.get('user_name')
        user_id = request.cookies.get('user_id')
        login_time = request.cookies.get('login_time')
        logout_time = datetime.now()

        # Log user logout and session duration
        log_message = (f"User '{user_name}' (ID: {user_id}) logged out. "
                       f"Session started at {login_time} and ended at {logout_time}.")
        app.logger.info(log_message, extra={'user_id': user_id, 'user_name': user_name})

    # Create a response to redirect to the login page
    resp = make_response(redirect(url_for('login')))
    # Clear the 'logged_in' cookie with secure attributes
    resp.set_cookie('logged_in', '', expires=0, httponly=True, secure=True, samesite='Lax')
    # Flash a message to the user
    flash('You have been logged out.')
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
