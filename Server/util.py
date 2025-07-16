import joblib
import os
import json
import numpy as np
import base64
import cv2
import pywt

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_saved_artifacts():
    """Initializes required dicitonaries and model

    Args: none

    Returns: none
    """

    print("Loading artifacts... ")
    global __class_name_to_number
    global __class_number_to_name

    with open(BASE_DIR + "/artifacts/class_dictionary.json", 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(BASE_DIR + '/artifacts/saved_BW_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("Done")

#converts class num to name
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_base64_for_test_srk():
    """Gets a base64 encoded string for a specific test image. Used for testing only
    """
    with open(BASE_DIR + "/b64_test.txt") as f:
        return f.read()

def get_cv2_image_from_base64_string(b64str):
    """ Credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library

    Args: b64str: A base64 string representing an image

    Returns: img: A cv2 formatted image
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_images(image_path, image_base64_data):
    """This is a slightly modified version of the get_cropped_image() function from the model notebook

    Args: image_path: A path to an image 
          image_base64_data: A base64 string representing an image

    Returns: cropped_faces: a list containing the cropped faces of anyone recognized in the image
    """

    face_cascade = cv2.CascadeClassifier(BASE_DIR + '/opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(BASE_DIR + '/opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

# This is the same wavelet transormation function created in the model notebook.
# When given an image, it returns the wavelet format of the image 

def wavelet_transform(img, mode='haar', level=1):
    """This is the same wavelet transormation function created in the model notebook.

    Args: img: A cv2 formatted image 
          mode: Specifies the wavedec2 function mode (leave default)
          level: specifies thewavedec2 function level (leave default)
    
    Returns: img_wav: a wavelet transformed image
    """
    #convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY )
    #convert to float
    img =  np.float32(img)   
    img /= 255;
    # compute coefficients 
    coeffs = pywt.wavedec2(img, mode, level=level)

    #Process Coefficients
    coeffs_H = list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    img_wav = pywt.waverec2(coeffs_H, mode);
    img_wav *= 255;
    img_wav =  np.uint8(img_wav)

    return img_wav



def classify_image(image_base64, image_path=None):
    """Stacks the cropped raw image and the wavelet image and runs it through the prediction model

    Args: image_path: A path to an image 
          image_base64: A base64 string representing an image

    Returns: result: A dictionary containing the predicted class, probabilities, and the name to number dictionary
    """

    imgs = get_cropped_images(image_path, image_base64)

    result = []
    for img_raw in imgs:
        scaled_raw = cv2.resize(img_raw, (32, 32))
        img_wav = wavelet_transform(img_raw, 'db1', 5)
        scaled_wav = cv2.resize(img_wav, (32,32))
        stacked_img = np.vstack((scaled_raw.reshape(32*32*3,1), scaled_wav.reshape(32*32,1)))

        len_image_array = 32*32*3 + 32*32

        final = stacked_img.reshape(1,len_image_array).astype(float)

        #runs prediction and stores resulting data
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

if __name__ == '__main__':
    load_saved_artifacts()
    #test predictions
    #base64 string test
    print("Testing srk_test1")
    print(classify_image(get_base64_for_test_srk(), None))
    #file tests
    print("\nTesting hrithik_roshan_test1")
    print(classify_image(None, BASE_DIR + "/test_images/hrithik_roshan_test1.png"))
    print("\nTesting kareena_kapoor_test1")
    print(classify_image(None, BASE_DIR + "/test_images/kareena_kapoor_test1.jpg"))
    print("\nTesting deepika_padukone_test1")
    print(classify_image(None, BASE_DIR + "/test_images/deepika_padukone_test1.jpg"))
    print("\nTesting deepika_and_srk_test")
    print(classify_image(None, BASE_DIR + "/test_images/deepika_and_srk_test.jpg"))
    

#h_r test fails and misclassifies as srk (49.25 to 49.48)
