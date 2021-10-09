from flask import Blueprint, request

predict_bp = Blueprint("predict", __name__)
'''
@predict_bp.route("/predict")
def get_text():
    text = request.args.get("text", None)
    text_list = [text]

    if text is None:
        return "No text given", 400
    elif text
'''