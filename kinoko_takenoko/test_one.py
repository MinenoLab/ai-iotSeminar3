# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import numpy as np

# ネットワーク定義
class CNN(chainer.Chain):
    def __init__(self, n_out):
        w = I.Normal(scale=0.05)#重みの初期化
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), #畳み込み層、1層目
            conv2=L.Convolution2D(16, 32, 5, 1, 0),#畳み込み層、2層目
            conv3=L.Convolution2D(32, 64, 5, 1, 0),#畳み込み層、3層目
            l4=L.Linear(None, n_out, initialW=w), #全結合層
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)  #プーリング層、1層目
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) #プーリング層、2層目
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) #プーリング層、3層目
        return self.l4(h3)

def main():
     # オプション処理
    parser = argparse.ArgumentParser(description='きのこの山、たけのこの里判別(1枚のテストデータ)')
    parser.add_argument('--model', '-m', type=str, default="CNN.model",help='model file name')#モデルデータのパス
    parser.add_argument('--data', '-d', default='data/test/takenoko_0045.jpg',#テスト画像のパス
                        help='location of image data')
    args = parser.parse_args()
    
    #画像読み込み
    im = Image.open(args.data)
    width, height = im.size
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, width-1, height-1), fill=(None), outline=(0, 255, 0))
    data = np.asarray(im,dtype=np.float32)
    data = data.transpose(2,0,1)

    # モデルから判別
    model = L.Classifier(CNN(2))
    
    chainer.serializers.load_npz(args.model, model)# モデル読み込み
    y = model.predictor(chainer.Variable(np.array([data])))#予測結果取得
    y = F.softmax(y).data#ソフトマックス関数において、予測確率取得
    pred_prob = y.max(axis=1)#予測確率取得
    pred_label = "takenoko" if y.argmax() == 1  else "kinoko"
    draw.text((2,2),pred_label+'['+str(round(float(pred_prob)*100,2))+'%]',fill=(0,100,0,255))#テスト画像に確率病害
    im.show()#描画結果表示
    
if __name__ == '__main__':
    main()

    

