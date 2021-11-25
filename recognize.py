from keras.models import model_from_json
import numpy as np
import cv2
import os
from PIL import Image
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.properties import ObjectProperty, StringProperty
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.config import Config
from kivy.graphics.texture import Texture
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
IMG_HEIGHT = 125
IMG_WIDTH = 125
Config.set('graphics', 'resizable', 0)
Builder.load_file('design.kv')

disease_classes = { 
    0:'Acne',
    1:'Hair Loss', 
    2:'Hives', 
    3:'Melanoma',
    4:'Nail fungus or other Nail Disease'
}


def load_model():

    # load json and create model
    json_file = open('cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('cnn_model.h5')
    print("Loaded model from disk")

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return loaded_model

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resize_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    final_img_array = np.array(resize_image).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    return final_img_array.astype('float32')/255

def read_img_from_camera(image):
    resize_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    final_img_array = np.array(resize_image).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    return final_img_array.astype('float32')/255

def predict_img(img, model):
    result = model.predict_proba(img)

    print("Final result is: ", result)

    return np.argmax(result), round(result[0][np.argmax(result)]*100,2)


class FileChoosePopup(Popup):
    load = ObjectProperty()


class MyLayout(Widget):
    file_path = StringProperty("No file chosen")
    the_popup = ObjectProperty(None)
    
    def open_popup(self):
        self.the_popup = FileChoosePopup(load=self.load)
        self.the_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.the_popup.dismiss()
        print(self.file_path)
        # check for non-empty list i.e. file selected
        if self.file_path:
            self.ids.image.source = self.file_path
            self.ids.imglabel.text = " "
            self.img = read_img(self.file_path)

    def open_camera(self):

        rawCapture = PiRGBArray(camera)
        # allow the camera to warmup
        time.sleep(0.1)
        # grab an image from the camera
        camera.capture(rawCapture, format="bgr")

        image = rawCapture.array

        buf1 = cv2.flip(image, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.ids.image.texture = image_texture
        self.ids.imglabel.text = " "

        self.img = read_img_from_camera(image)


    def classify_img(self):
        try:
            model = load_model()
            # img = read_img(self.file_path)
            prediction, probability = predict_img(self.img, model)
            detected_disease = str(disease_classes[prediction])
            self.ids.pred_label.text = detected_disease +" with "+ str(probability) +" probability."
            print(disease_classes[prediction])
        except Exception as e:
            print(e)

class SkinDiseaseApp(App):
    def build(self):
        Window.clearcolor = (1,1,1,1)
        Window.size = (430, 530)
        return MyLayout()


if __name__ == '__main__':
    SkinDiseaseApp().run()
