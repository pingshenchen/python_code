from flask import Flask,request,jsonify

app = Flask(__name__)


@app.route('/index',methods=['GET','POST'])
def index():
    age=request.args.get('age')
    print(age)


    xx=request.form.get('xx')
    print(xx)
    print(request.json)
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run()