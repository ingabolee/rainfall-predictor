import pandas
from sklearn import linear_model
from flask import Flask, request, render_template 



app = Flask(__name__)
df = pandas.read_csv("file.csv")
X = df[['Evaporation', 'Sunshine', 'WindSpeed', 'Humidity', 'Pressure', 'Temperature']]
y = df['Rainfall']

regr = linear_model.LinearRegression()
regr.fit(X.values, y.values)

@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       Evaporation = request.form.get("Evaporation")
       Sunshine = request.form.get("Sunshine")
       WindSpeed = request.form.get("WindSpeed")
       Humidity = request.form.get("Humidity")
       Pressure = request.form.get("Pressure")
       Temperature = request.form.get("Temperature")

       predictedRain = regr.predict([[Evaporation, Sunshine, WindSpeed, Humidity, Pressure, Temperature]])

       
       return f"Rainfall amount is {predictedRain}"
    return render_template("form.html")
  
if __name__=='__main__':
   app.run()
