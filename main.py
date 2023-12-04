import torch
from flask import Flask, render_template, request

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import random
import json

app = Flask(__name__)

# Load the data and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

try:
    data = torch.load('data.pth')

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(data["model_state"])
    model.eval()
except FileNotFoundError:
    print("Please train the model and save the data.pth file.")

# Function to get a response from the chatbot
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    with torch.no_grad():
        output = model(torch.from_numpy(X))
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."

# Default route to render the initial page
@app.route("/")
def home():
    return render_template("base.html", user_message="", bot_response="")

# Flask route for handling chat interactions
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["user_message"]
    bot_response = get_response(user_message)
    return render_template("base.html", user_message=user_message, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
