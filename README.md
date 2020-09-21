# MASK_DETECTOR_PYTHON_APP
This is a python app that can detect whether a person is wearing a mask or not.
## Dependencies:
- Opencv
- keras
- Tensorflow
- Dataset: obtained from google
## Observation:
- To detect whether a person or a group of people in a given image is wearing a mask or not ,we can use the script face_detector.py(use command : python face_detector.py)
- For video analysis we can use the web cam's live feed using the script mask_detector.py(use command:python mask_detector.py)
- The CNN model i have trained is named as model1.h5 and the code for training the model is given in MaskDetector.ipynb and accuracy of the model is 0.9691(validation accuracy)
- the required generated output files are named as output3.jpg and output2.jpg
![alt text](https://github.com/subhamChakraborty23/MASK_DETECTOR_PYTHON_APP/blob/master/test2.jpg)
![alt text](https://github.com/subhamChakraborty23/MASK_DETECTOR_PYTHON_APP/blob/master/detected_o2.jpg)
![alt text](https://github.com/subhamChakraborty23/MASK_DETECTOR_PYTHON_APP/blob/master/test1.jpg)
![alt text](https://github.com/subhamChakraborty23/MASK_DETECTOR_PYTHON_APP/blob/master/output3.jpg)


## Improvement:
This model's accuracy can be increased by using the following ways:
- Using a more better labelled and huge dataset,
- Using a FasterRCNN model,
- Using a pretrained model and better transfer learning methods the models accuracy can be increased and the detection will become more generalised.


