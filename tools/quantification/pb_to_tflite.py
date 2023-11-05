import sys
import tensorflow.compat.v1 as tf
import numpy as np
import pathlib
import cv2


def representative_dataset_gen():
    for jpg in pathlib.Path(sys.argv[1]).glob('*.jpg'):
        img = cv2.imread(jpg.as_posix())
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        input = cv2.resize(input, (56, 56))
        input = input[np.newaxis, :, :, :]
        input = input/255.
        yield [input.astype(np.float32)]


def quantize(model_path, quantized_model_path):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        model_path, ["Input"], ["Identity"], {"Input": [1, 56, 56, 3]})
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    quantized_tflite_model = converter.convert()
    open(quantized_model_path, "wb").write(quantized_tflite_model)


quantize(sys.argv[2], sys.argv[3])
