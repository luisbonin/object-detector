Clone TensorFlow models project from Git in a path of your preference that we will call WORKSPACE:
```
git clone https://github.com/tensorflow/models.git
```
Add TensorFlow research models to your Python path:
```
export PYTHONPATH=$PYTHONPATH:<WORKSPACE>/models:<WORKSPACE>/models/slim
```
Based on:
https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e