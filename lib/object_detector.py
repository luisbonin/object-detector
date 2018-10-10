import numpy
import tensorflow

from lib import file_manager
from lib.constants import DOWNLOAD_BASE_URL, MODEL_NAME, MODEL_EXTENSION, FROZEN_GRAPH, FROZEN_GRAPH_EXTENSION, \
                          PATH_TO_LABELS, TEST_IMAGE_PATHS, PATH_TO_RESULTS
from object_detection.utils import label_map_util, ops, visualization_utils
from matplotlib import pyplot
from PIL import Image


class ObjectDetector(object):
    def __init__(self, url=DOWNLOAD_BASE_URL, model_name=MODEL_NAME, frozen_graph=FROZEN_GRAPH):
        self.model_file = model_name + MODEL_EXTENSION
        self.url = url + self.model_file
        self.frozen_graph = frozen_graph + FROZEN_GRAPH_EXTENSION
        self.frozen_graph_path = model_name + '/' + self.frozen_graph
        self.category_index = None
        self.detection_graph = None

    def download_model(self):
        file_manager.download(url=self.url, filename=self.model_file)
        file_manager.extract(filename=self.model_file, specific_file=self.frozen_graph)

    def load_model(self):
        self.detection_graph = tensorflow.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()

            with tensorflow.gfile.GFile(self.frozen_graph_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')

        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    def run_inference_for_images(self):
        self.download_model()
        self.load_model()

        for index, image_path in enumerate(TEST_IMAGE_PATHS):
            image = Image.open(image_path)

            # The array based representation of the image will be used later in order to prepare the result image with
            # boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            numpy.expand_dims(image_np, axis=0)

            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)

            # Visualization of the results of a detection.
            visualization_utils.visualize_boxes_and_labels_on_image_array(image_np,
                                                                          output_dict['detection_boxes'],
                                                                          output_dict['detection_classes'],
                                                                          output_dict['detection_scores'],
                                                                          self.category_index,
                                                                          instance_masks=output_dict.get(
                                                                              'detection_masks'),
                                                                          use_normalized_coordinates=True,
                                                                          line_thickness=8)
            pyplot.imsave(PATH_TO_RESULTS + '/' + 'result_image_{}'.format(index), image_np)

    def __get_class(self, index):
        return self.category_index[index]['name']

    @staticmethod
    def __get_image_size(image):
        return image.shape[0], image.shape[1]

    @staticmethod
    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tensorflow.Session() as session:
                # Get handles to input and output tensors
                operations = tensorflow.get_default_graph().get_operations()
                all_tensor_names = {output.name for operation in operations for output in operation.outputs}
                tensor_dict = {}

                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                            'detection_masks']:
                    tensor_name = key + ':0'

                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tensorflow.get_default_graph().get_tensor_by_name(
                            tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tensorflow.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tensorflow.squeeze(tensor_dict['detection_masks'], [0])

                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image
                    # size.
                    real_num_detection = tensorflow.cast(tensor_dict['num_detections'][0], tensorflow.int32)
                    detection_boxes = tensorflow.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tensorflow.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = ops.reframe_box_masks_to_image_masks(detection_masks,
                                                                                    detection_boxes,
                                                                                    image.shape[0],
                                                                                    image.shape[1])
                    detection_masks_reframed = tensorflow.cast(tensorflow.greater(detection_masks_reframed, 0.5),
                                                               tensorflow.uint8)

                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tensorflow.expand_dims(detection_masks_reframed, 0)

                image_tensor = tensorflow.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = session.run(tensor_dict, feed_dict={image_tensor: numpy.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(numpy.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    @staticmethod
    def load_image_into_numpy_array(image):
        (image_width, image_height) = image.size

        return numpy.array(image.getdata()).reshape((image_height, image_width, 3)).astype(numpy.uint8)
