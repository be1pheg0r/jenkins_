pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }
        stage('Fetch Data') {
            steps {
                sh '''
                . venv/bin/activate
                ./venv/bin/python3 files/fetch_data.py
                '''
            }
        }
        stage('Preprocess Data') {
            steps {
                sh '''
                . venv/bin/activate
                ./venv/bin/python3 files/preprocess.py
                '''
            }
        }
        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                ./venv/bin/python3 files/train.py
                '''
            }
        }
        stage('Deploy Model') {
            steps {
                sh '''
                . venv/bin/activate
                ./venv/bin/python3 -m uvicorn files.app:app --reload
                '''
            }
        }
    }
}