from flask import Flask, jsonify, make_response, render_template, request, make_response
import json
import uuid
import os
import random
import pickle
import time
from settings import langs_ace, langs

app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = False
app._static_folder = os.path.abspath("templates/static/")

model_path = 'models/bayes_m1.sav'
model = None

@app.route("/", methods=["GET"])
def index():
	# load the model
	global model
	model = pickle.load(open(model_path, 'rb'))

	# write javascript
	with open("language.txt", "w") as file:
		file.write("ace/mode/javascript")

	# load the main page of interface
	return render_template("layouts/index.html")


@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
	jsdata = request.form['txt_data']
	txtstring = json.loads(jsdata)

	# call model to compute language from string
	# unfortunately, this takes ~ 6 seconds
	r = model.predict([txtstring])[0]

	# set the language mode and write
	print("real language ", langs[r])
	print("ace language ", langs_ace[r])
	language_txt = "ace/mode/{}".format(langs_ace[r])
	with open("language.txt", "w") as file:
		file.write(language_txt)

	return jsdata

@app.route('/getmethod')
def get_javascript_data():
	#supported list
	#https://cloud9-sdk.readme.io/docs/language-mode
	with open("language.txt", "r") as file:
		langtext = file.read()
	#it is a simple string no special characters so no need to json it
	print ("selecting language:", langtext)
	return langtext

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)