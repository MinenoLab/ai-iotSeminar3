# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import numpy as np
import shutil

# ネットワーク定義
class CNN(chainer.Chain):
    def __init__(self, n_out):
        w = I.Normal(scale=0.05)#重みの初期化
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), #畳み込み層、1層目
            conv2=L.Convolution2D(16, 32, 5, 1, 0),#畳み込み層、2層目
            conv3=L.Convolution2D(32, 64, 5, 1, 0),#畳み込み層、3層目
            l4=L.Linear(None, n_out, initialW=w),  #全結合層
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)) , ksize=2, stride=2) #プーリング層、1層目
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) #プーリング層、2層目
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) #プーリング層、3層目
        return self.l4(h3)

#混同行列作成
def print_cmx(y_true, x_pred,image):
    cm = confusion_matrix(y_true, x_pred)
    plt.figure(figsize=(7,5))#描画用ウィンドウ作成
    classNames = ['Negative\n(label: kinoko)','Positive\n(label: takenoko)']#ラベル定義
    plt.title('Counfusion Matrix - Test data')# タイトル
    plt.ylabel('True label')# y軸のラベルのタイトル
    plt.xlabel('Predicted label')# x軸のラベルのタイトル
    tick_marks = np.arange(len(classNames))# ラベルを配置する為の情報
    plt.xticks(tick_marks, classNames)    # x軸ラベルを作成
    plt.yticks(tick_marks, classNames)    # y軸ラベルを作成
    s = [['TN','FP'], ['FN', 'TP']]# 分類結果の指標
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))#分類結果を作成
    plt.colorbar()# カラーバー
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)#描画
    plt.savefig(os.path.join(image,'confusin_matrix_test.png'))# 保存

        
def main():
     # オプション処理
    parser = argparse.ArgumentParser(description='きのこの山、たけのこの里判別(全テストデータ)')
    parser.add_argument('--model', '-m', type=str, default="CNN.model",help='model file name')#モデルデータのパス
    parser.add_argument('--outdir', '-o', default='result',help='output directory')# 混同行列等の保存先
    parser.add_argument('--datadir', '-d', default='data/test',help='test data directory')#テストデータのディレクトリのパス
    args = parser.parse_args()

    #画像読み込み
    test = []
    for c in os.listdir(args.datadir):
        label = 'kinoko' if 'kinoko' in c else 'takenoko'
        data = Image.open(os.path.join(args.datadir, c))
        data = np.asarray(data,dtype=np.float32)
        data = data.transpose(2,0,1)
        test.append([data,label, os.path.join(args.datadir,c), c])
        
    # 分類
    model = L.Classifier(CNN(2))
    y_true=[] #正解ラベル
    x_pred=[] #予測ラベル
    chainer.serializers.load_npz(args.model,model)# モデル読み込み
    for data , label ,path, name  in  test:
        y = model.predictor(chainer.Variable(np.array([data])))#予測結果取得
        y = F.softmax(y).data #ソフトマックス関数において、予測確率取得
        c = y.argmax()#予測ラベル取得
        y_true.append((1 if 'takenoko' in label else 0))
        x_pred.append(c)
        print(name+
              ' was predicted as '+
              ('takenoko' if c==1 else 'kinoko  ')+
              ' with a '+str(round(float(y.max(axis=1))*100,2))+
              '% possibility')# 分類結果のコンソール出力
        shutil.copy(path, "./data/pred/"+('takenoko/' if c==1 else 'kinoko/')+name)# フォルダに振り分け
    #判別結果
    print_cmx(y_true, x_pred, args.outdir)
    
if __name__ == '__main__':
    main()

