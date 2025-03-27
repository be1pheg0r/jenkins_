pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }
        stage('Fetch Data') {
            steps {
                bat 'python files/fetch_data.py'
            }
        }
        stage('Preprocess Data') {
            steps {
                bat 'python files/preprocess.py'
            }
        }
        stage('Train Model') {
            steps {
                bat 'python files/train.py'
            }
        }
        stage('Deploy Model') {
            steps {
                bat 'start /b python files/app.py'
            }
        }
    }
}