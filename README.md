# Online Fraud Payment Detection using Balanced ML Algorithms

Django-based online payment fraud detection project using machine-learning algorithms and SMOTE balancing.

## Python environment

Tested during the cleanup with Python 3.14.6 and Django 6.1.1.

## Setup

1. Create a virtual environment:
   `py -3.14 -m venv venv314`
2. Install dependencies:
   `venv314\\Scripts\\python.exe -m pip install -r requirements.txt`
3. Apply database migrations:
   `venv314\\Scripts\\python.exe manage.py migrate`
4. Start the development server:
   `venv314\\Scripts\\python.exe manage.py runserver`
5. Open `http://127.0.0.1:8000/`.

## Main project files

- `Fraud/` - Django project configuration
- `FraudApp/` - application code, templates, CSS and images
- `Dataset/` - fraud-detection training dataset and sample test data
- `model/data.npy` - saved training data used by the original ML workflow
- `db.sqlite3` - SQLite database used by the application
- `manage.py` - Django management entry point

The project package intentionally excludes virtual environments, Python cache files, unrelated personal documents, duplicate datasets, and temporary files.
