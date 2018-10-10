from lib import object_detector


if __name__ == '__main__':
    object_detector = object_detector.ObjectDetector()
    object_detector.run_inference_for_images()
