# Flask 기본 라이브러리 import
from flask import Flask, render_template

# Flask 앱 인스턴스 생성
app = Flask(__name__)


# 기본 라우트 설정
@app.route("/")
def home():
    # 메인 페이지 간단 응답
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


# 앱 실행 (직접 실행 시)
if __name__ == "__main__":
    app.run(debug=True)
