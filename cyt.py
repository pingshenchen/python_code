import numpy as np
with open(r'/Users/mac/PycharmProjects/python_code/pythonProject/deeplearning/DNN/ml2021spring-hw1/covid.train.csv', 'r') as f:
    train_data = f.readlines()
    train_data = [line.split('\n') for line in train_data][1:]  # 分行之后不要第一行
    train_data = [each[0].split(',') for each in train_data]  # 对于每一行 去掉后面的空格
    train_data = np.array(train_data)  # 转换成numpy的矩阵

    train_x = train_data[:, 1:-1]  # x是数据，y是标签 。第一个冒号表示所有行，第二个冒号表示
    train_y = train_data[:, -1]  # 列。所以x就是第2列到倒数第二列。y就是倒数第一列。