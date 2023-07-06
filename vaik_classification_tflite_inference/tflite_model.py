from typing import List, Dict, Tuple
import multiprocessing
import platform
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite


class TfliteModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None, num_thread: int = None):
        self.classes = classes
        num_thread = multiprocessing.cpu_count() if num_thread is None else num_thread
        self.__load(input_saved_model_path, num_thread)

    def inference(self, input_image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        resized_image_array = self.__preprocess_image(input_image, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image_array)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __load(self, input_saved_model_path: str, num_thread: int):
        try:
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, num_threads=num_thread)
            self.interpreter.allocate_tensors()
        except RuntimeError:
            _EDGETPU_SHARED_LIB = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'
            }[platform.system()]
            delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, experimental_delegates=delegates,
                                                  num_threads=num_thread)
            self.interpreter.allocate_tensors()
        self.model_input_shape = self.interpreter.get_input_details()[0]['shape']

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]), dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize((int(resize_size[0]), int(resize_size[1])))
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image

    def __inference(self, resized_image: np.ndarray) -> np.ndarray:
        if len(resized_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resized_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resized_image.dtype}')
        self.__set_input_tensor(resized_image)
        self.interpreter.invoke()
        raw_pred = self.__get_output_tensor()[0]
        return raw_pred

    def __output_parse(self, pred: np.ndarray) -> Dict:
        pred_index = np.argsort(-pred)
        output_dict = {'score': pred[0][pred_index[0]].tolist(),
                       'label': [self.classes[class_index] for class_index in pred_index[0]]}
        return output_dict

    def __set_input_tensor(self, image: np.ndarray):
        input_tensor = self.interpreter.tensor(self.interpreter.get_input_details()[0]['index'])()
        input_tensor.fill(0)
        input_image = image.astype(self.interpreter.get_input_details()[0]['dtype'])
        input_tensor[0, :input_image.shape[0], :input_image.shape[1], :input_image.shape[2]] = input_image

    def __get_output_tensor(self) -> List[np.ndarray]:
        output_details = self.interpreter.get_output_details()
        output_tensor = []
        for index in range(len(output_details)):
            output = self.interpreter.get_tensor(output_details[index]['index'])
            scale, zero_point = output_details[index]['quantization']
            if scale > 1e-4:
                output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor