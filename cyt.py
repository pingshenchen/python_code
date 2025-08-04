import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import flask
from gensim.models import Word2Vec
import jieba
sentences = [
    list(jieba.cut('我喜歡吃蘋果')),
    list(jieba.cut('蘋果很好吃')),
    list(jieba.cut('水果是健康的')),
    list(jieba.cut('梨子也很好吃')),
    list(jieba.cut('我也喜歡吃柳丁')),
    list(jieba.cut('蘋果柳丁都是一種水果')),
    list(jieba.cut('蘋果是一種又香又甜的水果')),
    list(jieba.cut('梨子跟柳丁也是一種又香又甜的水果')),
]

# 訓練詞向量模型
model = Word2Vec(sentences, window=5, min_count=1, workers=4)

# 獲取所有詞
vocab = model.wv.index_to_key

# 獲取所有詞向量
vectors = model.wv[vocab]

print(vectors)