import numpy
from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import joblib


app = Flask(__name__)

@app.route('/')

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/rain')
def rain():
    #iris = load_iris()
    #model = GradientBoostingClassifier(n_estimators=50,  learning_rate=1, max_features=2, max_depth=2, random_state=0)
    #X_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    #model.fit(X_train,y_train)
    #model.predict(x_test)
    #pickle.dump(model,open("iris.pkl", "wb"))

    return render_template("rain.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    #loading the model
    rain_gb_model_path= 'C:\\Users\\Admin\\RainClassification_Prediction'
    rain_gb_model = joblib.load(rain_gb_model_path)

    maximum_temperature = request.form['maximum_temperature']
    minimum_temperature = request.form['minimum_temperature']
    wind_speed = request.form['wind_speed']
    wind_direction = request.form['wind_direction']
    rainfall_rate = request.form['rainfall_rate']

    #predicting using the model
    result = rain_gb_model.predict([[maximum_temperature, minimum_temperature, wind_speed, wind_direction, rainfall_rate]])

    return render_template("result.html", result = result[0])

@app.route('/casualties')
def casualties():
    return render_template('casualties.html')

@app.route("/casualties_predict", methods=["GET", "POST"])
def casualties_predict():

    #loading the model
    casualties_gb_model_path= 'C:\\Users\\Admin\\CasualtiesClassification_Prediction'
    casualties_gb_model = joblib.load(casualties_gb_model_path)

    population = request.form['population']
    population_density = request.form['population_density']
    precipitation_description = request.form['precipitation_description']
    distance_main = request.form['distance_main']
    distance_second = request.form['distance_second']
    site_elevation = request.form['site_elevation']
    site_slope = request.form['site_slope']

    #predicting using the model
    result = casualties_gb_model.predict([[population, population_density, precipitation_description, distance_main, distance_second, site_elevation, site_slope]])

    return render_template("casualtiesResult.html", result = result[0])

if __name__ == "__main__":
    app.run(debug=True)

