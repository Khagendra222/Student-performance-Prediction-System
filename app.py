from flask import Flask, render_template, request
import joblib

app = Flask(
    __name__,
    template_folder="flask_app/templates",
    static_folder="flask_app/static"
)

# Load model and encoders
model = joblib.load("models/student_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.form.to_dict()

        # Encode categorical values
        categorical_cols = [
            "school", "sex", "address", "famsize", "Pstatus",
            "Mjob", "Fjob", "reason", "guardian",
            "schoolsup", "famsup", "paid", "activities",
            "nursery", "higher", "internet", "romantic"
        ]

        for col in categorical_cols:
            data[col] = encoders[col].transform([data[col]])[0]

        # Create feature list in EXACT training order
        features = [
            data["school"],
            data["sex"],
            int(data["age"]),
            data["address"],
            data["famsize"],
            data["Pstatus"],
            int(data["Medu"]),
            int(data["Fedu"]),
            data["Mjob"],
            data["Fjob"],
            data["reason"],
            data["guardian"],
            int(data["traveltime"]),
            int(data["studytime"]),
            int(data["failures"]),
            data["schoolsup"],
            data["famsup"],
            data["paid"],
            data["activities"],
            data["nursery"],
            data["higher"],
            data["internet"],
            data["romantic"],
            int(data["famrel"]),
            int(data["freetime"]),
            int(data["goout"]),
            int(data["Dalc"]),
            int(data["Walc"]),
            int(data["health"]),
            int(data["absences"]),
            int(data["G1"]),
            int(data["G2"])
        ]

        prediction = model.predict([features])[0]

        if prediction == 1:
            result = "✅ PASS"
        else:
            result = "❌ FAIL"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)