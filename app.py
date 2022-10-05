from flask import Flask,render_template,request
import pandas as pd
import pickle
import datetime

app=Flask(__name__)
df=pd.read_csv("BangloreHouseData.csv")
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    locations=sorted(df["location"].unique())
    return render_template("index.html",locations=locations)

@app.route("/predict",methods=["GET","POST"])

def predict():
    inputs=[i for i in request.form.values()]
    predvalue=[int(i) for i in inputs[1:]]
    print(predvalue)
    newdf=df[df["location"]==inputs[0]]
    x=newdf.iloc[:,1:5]
    y=newdf.iloc[:,5]
    model.fit(x,y)
    k=model.predict([predvalue])[0]
    now=datetime.datetime.now()

    return render_template("submit.html",location=inputs[0],bhk=inputs[1],sqft=inputs[2],bath=inputs[3],upst=inputs[4],date_time=now,pred_val="₹ "+str(int((k))))


if __name__=='__main__':
    app.run(debug=True)


