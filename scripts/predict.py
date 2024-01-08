import pickle
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

OUTPUT_FILE = "../model/rf.bin"


print("Load model...")
with open(OUTPUT_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)


@app.route("/predict", methods=["POST"])
def predict():
    user = request.get_json()

    X = dv.transform([user])
    y_pred = model.predict(X)

    result = {"predicted_amount": float(y_pred)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
