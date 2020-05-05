import sys
import os
import numpy as np
import cv2
import tensorflow as tf
'''
https://docs.microsoft.com/ja-jp/azure/cognitive-services/custom-vision-service/export-model-python
OpenCvでのカメラ処理を前提に修正
'''

# 関数(チュートリアルから流用)--------------------------------------------------------
def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)


# 映像の処理-------------------------------------------------------------
def generateVideo(frame, network_input_size):
    h, w = frame.shape[:2]
    # downsizing
    if (h > 1600 and w > 1600):
        new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
        frame =  cv2.resize(frame, new_size, interpolation = cv2.INTER_LINEAR)
    # get center
    min_dim = min(w,h)
    max_square_image = crop_center(frame, min_dim, min_dim)
    frame = resize_to_256_square(max_square_image)
    # resizing
    frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_LINEAR)
    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(frame, network_input_size, network_input_size)
    return augmented_image

# グラフ作成-------------------------------------------------------------
def makeGraph(filename, labels_filename):
    # モデルとタグを読み込み-------------------------------------------------------------
    graph_def = tf.compat.v1.GraphDef()
    labels = []

    # Import the TF graph
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    # Get the input size of the model
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]

    return graph_def, labels, network_input_size

# グラフ作成-------------------------------------------------------------
def inferenceGraph(augmented_image, labels):
    # 推論処理-------------------------------------------------------------
    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)
        # 結果確認-------------------------------------------------------------
        # Print the highest probability label
        highest_probability_index = np.argmax(predictions)
        caption = ('Classified as: ' + labels[highest_probability_index])
        return caption
        # Or you can print out all of the results mapping labels to probabilities.
        '''
        label_index = 0
        for p in predictions:
            truncated_probablity = np.float64(np.round(p,8))
            print (labels[label_index], truncated_probablity)
            label_index += 1
        '''


# メイン処理-------------------------------------------------------------
def main():
    # These are set to the default names from exported models, update as needed.
    base_path = os.getcwd()
    filename = f"{base_path}/exported_tf/model.pb"
    labels_filename = f"{base_path}/exported_tf/labels.txt"

    # この時点で計算グラフは生成されている
    graph_def, labels, network_input_size = makeGraph(
        filename, labels_filename)

    # openVideo
    try:
        capture = cv2.VideoCapture(0)
        while(True):
            ret, frame = capture.read()
            if ret is False:
                print('I could not found camera stram')
                break
            augmented_image = generateVideo(frame, network_input_size)
            caption = inferenceGraph(augmented_image, labels)
            frame = cv2.putText(
                frame, caption, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50, 60, 80), 2
                )
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    except:
        print("Error:", sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info()[2]))


if __name__ == '__main__':
    main()
