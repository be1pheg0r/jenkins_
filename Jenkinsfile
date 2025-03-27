pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Fetch Data') {
            steps {
                sh 'python files/fetch_data.py'
            }
        }
        stage('Preprocess Data') {
            steps {
                sh 'python files/preprocess.py'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python files/train.py'
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'start /b uvicorn files.app:app --reload'
            }
        }
    }
}