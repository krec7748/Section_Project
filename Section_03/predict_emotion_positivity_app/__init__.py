import os, sys
from flask import Flask, request, jsonify, render_template
import pickle

# CSV 파일 경로와 임시 파일 경로입니다.
#CSV_FILEPATH = os.path.join(os.getcwd(), __name__, 'users.csv') 
#TMP_FILEPATH = os.path.join(os.getcwd(), __name__, 'tmp.csv') 

def create_app():
    app = Flask(__name__)
    
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from predict_emotion_positivity_app.views.main_views import main_bp, predict_bp, db_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(db_bp)

    return app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True)