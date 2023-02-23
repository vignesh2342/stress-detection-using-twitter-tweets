from flask import Flask, render_template, url_for, request, redirect
import pickle
from sklearn.feature_extraction.text import CountVectorizer


# Load the model from the file
with open('./stress_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the CountVectorizer from the file
with open('./stress_cv.pkl', 'rb') as file:
    cv = pickle.load(file)

# Create a new instance of CountVectorizer using the same parameters as the original
new_cv = CountVectorizer(vocabulary=cv.vocabulary_)

# Create some new data to make predictions on
# new_data = ["I'm feeling really stressed out today", "I'm feeling great today"]

# Transform the new data using the loaded CountVectorizer
# new_X = new_cv.transform(new_data)

# Use the loaded model to make predictions on the new data
# predictions = model.predict()

# print(predictions)  # Output: ['Stress', 'No Stress']


app = Flask(__name__)

# Load the model and CountVectorizer from the pickle files
# with open(r'./stress_model.pkl', 'rb') as file:
#     model = pickle.load(file)
# with open(r'./stress_cv.pkl', 'rb') as file:
#     cv = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/user_choice")
def user_choice():
    return render_template('user_choice.html')


@app.route('/tweets', methods=["GET", "POST"])
def tweets():
    return render_template('tweets.html')


@app.route('/tweets_results', methods=["GET", "POST"])
def tweets_results():
    text = request.form.get("text-input")
    print('Text: ', text)
    new_X = new_cv.transform([text])
    prediction = model.predict(new_X)[0]
    print('Testing: ', prediction)

    return render_template('tweets.html', pred=prediction)


@app.route("/cam")
def cam():
    return render_template('cam.html')


# @app.route("/cam_results")
# def cam_results():
#     # start()
#     return render_template('cam.html')


# @app.route("/tweets/")
# def tweets():
#     return render_template('tweets.html')
if __name__ == '_main_':
    app.run(debug=True)
