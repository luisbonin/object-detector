import os

# Model definitions
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_EXTENSION = '.tar.gz'
DOWNLOAD_BASE_URL = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection
FROZEN_GRAPH = 'frozen_inference_graph'
FROZEN_GRAPH_EXTENSION = '.pb'

# List of the strings that is used to add correct label for each box
PATH_TO_DATA = 'data'
LABEL_MAP = 'mscoco_label_map'
LABEL_MAP_EXTENSION = '.pbtxt'
PATH_TO_LABELS = os.path.join(PATH_TO_DATA, LABEL_MAP + LABEL_MAP_EXTENSION)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Path to results
PATH_TO_RESULTS = 'results'
