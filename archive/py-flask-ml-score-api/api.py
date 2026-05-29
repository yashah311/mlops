import logging
from joblib import load
from flask import Flask, request

INPUT_ARRAY = [[13.77,0.2344]]
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
clf = load('svc_model.model')

@app.route('/')
def hello_world():
   clf = load('svc_model.model')
   preds = clf.predict(INPUT_ARRAY)
   app.logger.info(" Inputs: {}".format(INPUT_ARRAY))
   app.logger.info(" Prediction: {}".format(preds))
   return str(preds)

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  app.logger.info("Record To predict: {}".format(data))
  app.logger.info(type(data))
  input_data = [data["data"]]
  app.logger.info(input_data)
  prediction = clf.predict(input_data)
  app.logger.info(prediction)
  response_data = prediction[0]
  return {"prediction": str(response_data)}


if __name__ == '__main__' :
	app.run(host='0.0.0.0', port=5050, debug=True)	
