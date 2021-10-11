import os, sys
from flask import Flask

def create_app():
    app = Flask(__name__)
    
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from predict_emotion_positivity_app.views.main_views import main_bp, predict_bp, db_bp, db_sentiment_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(db_bp)
    app.register_blueprint(db_sentiment_bp)

    return app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True)