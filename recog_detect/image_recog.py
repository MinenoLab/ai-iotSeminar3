#各種ライブラリのインポート
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import sys

#コマンドライン引数の取り出し（python3 image_recog.py xxx ←「xxx」の部分）
if len(sys.argv) != 2:                                      #引数（正確には異なる）が2個でないとき
    print("使い方: python3 image_recog.py [image_path]")    #プログラムの起動方法の提示
    sys.exit(1)                                             #プログラムの終了

#引数（上記のxxx部分）の取り出し（画像のパスがimg_pathに代入される）
img_path = sys.argv[1]

#事前学習済みモデルの読み込み
model = MobileNet(input_shape=None, weights='imagenet')

#画像の読み込み
img = image.load_img(img_path, target_size=(224,224))       #224×224にリサイズして読み込む
img_array = image.img_to_array(img)                         #画像データを配列型に変換する
img_array = np.expand_dims(img_array, axis=0)               #配列の0次元目に次元を追加する

#画像データの前処理
x = preprocess_input(img_array)

#物体の予測
preds = model.predict(x)

#予測結果の出力（可能性の高い上位3クラスと確率）
print("\n予測結果TOP3(名前、確率)")
result = decode_predictions(preds, top=3)[0]
for i in range(3):
    print('{}：{}'.format(i+1,result[i][1:]))
