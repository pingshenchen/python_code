from flask import Flask,request,jsonify
import hashlib
app = Flask(__name__)

def get_user_dict():
    user_dict = {}
    with open('db.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            token,name=line.split(',')
        user_dict[token]=name
    return user_dict

@app.route('/index',methods=['GET','POST'])
def index():
    '''
    指定用户授权授权访问网站权限
    :return:
    '''
    # token是否为空
    token=request.args.get('token')
    if not token:
        return jsonify({'status':400,'msg':'认证失败'})
    # token是否合法
    user_dict=get_user_dict()
    if token not in user_dict:
        return jsonify({'status': 400, 'msg': '认证失败'})

    xx=request.form.get('xx')
    print(xx)
    print(request.json)
    return jsonify({'status': 200, 'msg': user_dict})

if __name__ == '__main__':
    app.run()