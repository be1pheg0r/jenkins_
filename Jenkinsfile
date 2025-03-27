pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'pip3 install -r requirements.txt'
            }
        }
        stage('Fetch Data') {
            steps {
                sh 'python3 files/fetch_data.py'
            }
        }
        stage('Preprocess Data') {
            steps {
                sh 'python3 files/preprocess.py'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python3 files/train.py'
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'uvicorn files.app:app --reload'
            }
        }
    }
}