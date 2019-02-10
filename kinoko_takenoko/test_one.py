# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import numpy as np

# 畳み込みニューラルネットワークの定義
class CNN(chainer.Chain):
    def __init__(self, n_out):
        w = I.Normal(scale=0.05) # モデルパラメータの初期化
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), # 1層目の畳み込み層（フィルタ数は16）
            conv2=L.Convolution2D(16, 32, 5, 1, 0), # 2層目の畳み込み層（フィルタ数は32）
            conv3=L.Convolution2D(32, 64, 5, 1, 0), # 3層目の畳み込み層（フィルタ数は64）
            l4=L.Linear(None, n_out, initialW=w), #クラス分類用
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        # 9x9,64ch
        return self.l4(h3)


def main():
     # オプション処理
    parser = argparse.ArgumentParser(description='きのこの山、たけのこの里判別(1枚のテストデータ)')
    parser.add_argument('--model', '-m', type=str, default="CNN.model",
                        help='model file name')
    parser.add_argument('--data', '-d', default='data/test/takenoko/0045.jpg',
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
    model = L.Classifier(CNN(2))  # CNNにする
    chainer.serializers.load_npz(args.model, model)
    y = model.predictor(chainer.Variable(np.array([data])))
    y = F.softmax(y).data
    pred_prob = y.max(axis=1)
    pred_label = "takenoko" if y.argmax() == 1  else "kinoko"
    draw.text((2,2),pred_label+'['+str(round(float(pred_prob)*100,2))+'%]',fill=(0,100,0,255))

    #結果表示
    im.show()


if __name__ == '__main__':
    main()

    

