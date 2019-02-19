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

class CNN(chainer.Chain):
    def __init__(self, n_out):
        w = I.Normal(scale=0.05)
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), 
            conv2=L.Convolution2D(16, 32, 5, 1, 0),
            conv3=L.Convolution2D(32, 64, 5, 1, 0),
            l4=L.Linear(None, n_out, initialW=w), 
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) 
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        return self.l4(h3)

def print_cmx(y_true, x_pred,image):
    print('正解率(Accuracy)', accuracy_score(y_true, x_pred))
    print('適合率、精度(Precision)', precision_score(y_true, x_pred))
    print('再現率、検出率(Recall)', recall_score(y_true, x_pred))
    print('F値', f1_score(y_true, x_pred))

    cm = confusion_matrix(y_true, x_pred)
    plt.figure(figsize=(10,9))
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative\n(label: kinoko)','Positive\n(label: takenoko)']
    plt.title('Counfusion Matrix - Test data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.savefig(os.path.join(image,'confusin_matrix_test.png'))

    

def print_roc(y_true, x_pred,image):
    # FPR, TPR(, しきい値) を算出
    fpr, tpr, thresholds = roc_curve(y_true, x_pred)
    # AUC算出
    a = auc(fpr, tpr)
    # ROC曲線をプロット
    plt.figure() 
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%a)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(image,'roc_curve_auc.png'))

    
def main():
     # オプション処理
    parser = argparse.ArgumentParser(description='きのこの山、たけのこの里判別(全テストデータ)')
    parser.add_argument('--model', '-m', type=str, default="CNN.model",
                        help='model file name')
    parser.add_argument('--outdir', '-o', default='result',
                        help='output directory')
    parser.add_argument('--datadir', '-d', default='data/test',
                        help='test data directory')
    args = parser.parse_args()

    #画像読み込み
    test = []
    for c in os.listdir(args.datadir):
        label = 'kinoko' if 'kinoko' in c else 'takenoko'
        d = os.path.join(args.datadir, c)        
        imgs = os.listdir(d)
        for i in [f for f in imgs if ('jpg' in f)]:
            data = Image.open(os.path.join(d, i))
            data = np.asarray(data,dtype=np.float32)
            data = data.transpose(2,0,1)
            test.append([data,label, os.path.join(d,i)])

    # モデルから判別
    model = L.Classifier(CNN(2))
    y_true=[] #実際の
    x_pred=[] #モデルの識別結果
    chainer.serializers.load_npz(args.model,model)
    for data , label ,name in  test:
        #print(i.shape)
        y = model.predictor(chainer.Variable(np.array([data])))
        y = F.softmax(y).data
        c = y.argmax()
        y_true.append((1 if 'takenoko' in label else 0))
        x_pred.append(c)
        print(name+
              ' was predicted as '+
              ('takenoko' if c==1 else 'kinoko  ')+
              ' with a '+str(round(float(y.max(axis=1))*100,2))+
              '% possibility')

    #判別結果
    print_cmx(y_true, x_pred, args.outdir)
    print_roc(y_true, x_pred, args.outdir)


if __name__ == '__main__':
    main()

