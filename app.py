from flask import Flask,render_template,request
import pickle
import pandas as pd

data=pd.read_csv('players_21.csv')
with open("model.pkl","rb") as model_file:
     model=pickle.load(model_file)
app=Flask(__name__)

@app.route('/')
def player_position():
    return render_template('index.html')

@app.route('/input',methods=['GET','POST'])
def characterstics():
   if request.method=='POST':
      player_position=request.form['player_position']
      print(player_position)
   return render_template('input.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
   if request.method=='POST':
      Lrank=request.form['Lrank']
      print(Lrank)
      IR=request.form['IR']
      WF=request.form['WF']
      BT=request.form['BT']
      MR=request.form['MR']
      GKA=request.form['GKA']

      model=pickle.load(open('model.pkl','rb'))
      rating=model.predict([[float(Lrank),int(IR),int(WF),int(BT),int(MR),float(GKA)]])
      print(rating)

   return render_template('prediction.html',rating=rating)



if __name__=='__main__':
  app.run()

