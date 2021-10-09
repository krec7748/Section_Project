import csv

from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

"""
def import_data():
  with open(CSV_FILEPATH, newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    user_list = []
    for row in dataset:
      id_user = []
      id_user.append(int(row["id"]))
      id_user.append(row["username"])
      user_list.append(id_user)

  return user_list
"""

@main_bp.route('/')
def index():
    return render_template("index.html")