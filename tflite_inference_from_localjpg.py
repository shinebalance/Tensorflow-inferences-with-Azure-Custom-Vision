import os
import numpy as np
import cv2
# import tensorflow as tf
from tensorflow import lite as tflite
# import tflite_runtime.interpreter as tflite

'''
Guide:https://www.tensorflow.org/lite/convert/python_api?hl=ja
cd Py/tf_inference_withACS
pipenv shell
CV2処理で参考にしたリンク：https://qiita.com/iwatake2222/items/d63aa67e5c700fcea70a
'''


# 画像の準備-------------------------------------------------------------
# ヘルパー関数(いる？)
def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


# from jpg file to tensor
def convertJpg2tensor(imageFile, network_input_size, isShapePeep):
    # prepara input image
    image = cv2.imread(imageFile)
    image = resize_down_to_1600_max_dim(image)
    if isShapePeep is True:
        print('to 1600 >>>', image.shape)
    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w, h)
    image = crop_center(image, min_dim, min_dim)
    if isShapePeep is True:
        print('crop center >>>', image.shape)
    # Resize that square down to 256x256
    image = resize_to_256_square(image)
    if isShapePeep is True:
        print('resized >>>', image.shape)
    # Crop the center for the specified network_input_Size
    image = crop_center(image, network_input_size, network_input_size)
    image = image.reshape(
        1, image.shape[0], image.shape[1], image.shape[2])
    image_f32 = image.astype(np.float32)
    if isShapePeep is True:
        print('finally >>>', image_f32.shape)
    return image_f32


def main():
    # load from saved model
    base_path = os.getcwd()
    file = f"{base_path}/exported_savedmodel"
    # Get labels
    labels_filename = f"{file}/labels.txt"
    labels = []
    # list化しておいて最後の結果出力時に使う
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    # from converted tflite file
    tflite_model = 'exported_from_code.tflite'
    interpreter = tflite.Interpreter(model_path=tflite_model)

    # run inference
    # allocate
    interpreter.allocate_tensors()
    # get details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # get tensor from a jpg file
    imageFile = f"{base_path}/sample.jpg"
    NETWORK_INPUT_SIZE = 224
    image_f32 = convertJpg2tensor(
        imageFile, network_input_size=NETWORK_INPUT_SIZE, isShapePeep=False)
    # set tensor and invoke
    interpreter.set_tensor(input_details[0]['index'], image_f32)
    interpreter.invoke()
    # get_tensor() はテンソルのコピーを返す
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    # 結果の出力
    result = np.argmax(tflite_results[0])
    score = tflite_results[0][result]
    print("Classified >>>", labels[result], score)


if __name__ == '__main__':
    main()
