# app.py
from flask import Flask, render_template, request, jsonify
import os, pandas as pd
from model import run_forecast

app = Flask(__name__, static_folder='static', template_folder='templates')
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','외국인입국자_전처리완료_딥러닝용.csv'),encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df['년월'] = pd.to_datetime(df['연도'].astype(str)+'-'+df['월'].astype(str).str.zfill(2))
countries = sorted(df['국적'].unique())
purposes  = sorted(df['목적'].unique())

@app.route('/api/countries')
def api_countries(): return jsonify(['전체']+countries)
@app.route('/api/purposes')
def api_purposes(): return jsonify(['전체']+purposes)
@app.route('/')
def root(): return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    req = request.get_json()
    ci = req['combos'][0]['country']
    pi = req['combos'][0]['purpose']
    sy, sm = map(int, req['start_ym'].split('-'))
    ey, em = map(int, req['end_ym'].split('-'))
    sel_c = countries if ci=='전체' else [ci]
    results=[]
    for c in sel_c:
        ps = sorted(df[df['국적']==c]['목적'].unique()) if pi=='전체' else [pi]
        for p in ps:
            months=[]
            for y in range(sy, ey+1):
                if y==sy==ey: ms=range(sm,em+1)
                elif y==sy:     ms=range(sm,13)
                elif y==ey:     ms=range(1,em+1)
                else:           ms=range(1,13)
                months+=list(ms)
            res = run_forecast(c,p,sy,months)
            for item in res:
                results.append(item)
    return jsonify({'results':results})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
