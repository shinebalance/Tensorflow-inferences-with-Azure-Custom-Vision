import os
import tensorflow as tf
# from tensorflow import lite as tflite
# import tflite_runtime.interpreter as tflite

'''
convert from savedmodel
'''


# prepare tflite model------------------------------------------------------
# load from saved model
base_path = os.getcwd()
file = f"{base_path}/exported_savedmodel"

EXPORT_FILEPATH = "exported_from_code.tflite"

# with converter
# converter
converter = tf.lite.TFLiteConverter.from_saved_model(file)
tflite_model = converter.convert()
with open(EXPORT_FILEPATH, "wb") as f:
    f.write(tflite_model)

print(">>> Exported >>>", EXPORT_FILEPATH)
