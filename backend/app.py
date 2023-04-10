from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing import image
import keras.utils as image
from datetime import datetime
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import time
import numpy as np
from PIL import Image
from io import BytesIO
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'application/json'

# Load the model
path_to_model = './model_v1_inceptionV3.h5'
model = load_model(path_to_model)

# Define the categories
category = {
    0: ['burger', 'Burger'], 1: ['butter_naan', 'Butter Naan'], 2: ['chai', 'Chai'],
    3: ['chapati', 'Chapati'], 4: ['chole_bhature', 'Chole Bhature'], 5: ['dal_makhani', 'Dal Makhani'],
    6: ['dhokla', 'Dhokla'], 7: ['fried_rice', 'Fried Rice'], 8: ['idli', 'Idli'], 9: ['jalegi', 'Jalebi'],
    10: ['kathi_rolls', 'Kaathi Rolls'], 11: ['kadai_paneer', 'Kadai Paneer'], 12: ['kulfi', 'Kulfi'],
    13: ['masala_dosa', 'Masala Dosa'], 14: ['momos', 'Momos'], 15: ['paani_puri', 'Paani Puri'],
    16: ['pakode', 'Pakode'], 17: ['pav_bhaji', 'Pav Bhaji'], 18: ['pizza', 'Pizza'], 19: ['samosa', 'Samosa']
}

def predict_image(img_,model):
    # img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    
    index = np.argmax(prediction)
    # Return the processed image along with the predicted label
    return img_processed, category[index][1]

# Define the route to handle incoming image files
@app.route('/expiration_time', methods=['POST'])
def predict():
        # file = request.files['image']
        # filename = secure_filename(file.filename)
        # file.save(filename)

        # # Call the predict_image function to get the processed image and predicted label
        # img_processed, label = predict_image(filename, model)
        print(request.form['image_url'])
        # # Convert the processed image to a list and return the response
        # # response = {'label': label, 'image': img_processed.tolist()}
        # response={'label':label}
        # return response
        url = request.form['image_url']
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB') 
        img_processed, label = predict_image(img, model)
        response = {'label': label}
        return response

@app.route('/predict', methods=['POST'])
def expiration_time():
    data = request.get_json()
    preparation_time = datetime.strptime(data['preparation_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
    city = data['city']
    url = data['image_url']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB') 
    img_processed, label = predict_image(img, model)
    food_item = label.lower()
    weather = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=2705e4424ec88cbf001a26de8830ecea&units=metric").json()
    temperature = weather["main"]["temp"]
    city = weather["name"]

    food_items = {
        "burger": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "butter_naan": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "chapati": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "chole_bhature": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "dal_makhani": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "dhokla": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "fried_rice": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "idli": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "jalebi": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "kaathi_rolls": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "kadai_paneer": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "kulfi": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "masala_dosa": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "momos": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "pav_bhaji": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "pizza": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "samosa": {"0-20": 4, "20-30": 3, "30-40": 2, "40-50": 1},
        "chai": {"0-20": 3, "20-30": 1, "30-40": 1, "40-50": 1},
        "pani_puri": {"0-20": 3, "20-30": 1, "30-40": 1, "40-50": 1},
        "pakode": {"0-20": 3, "20-30": 1, "30-40": 1, "40-50": 1}
    }

    if temperature < 20:
        temp_range = "0-20"
    elif temperature >= 20 and temperature < 30:
        temp_range = "20-30"
    elif temperature >= 30 and temperature < 40:
        temp_range = "30-40"
    elif temperature >= 40 and temperature <= 50:
        temp_range = "40-50"
    
    expire_time = preparation_time + timedelta(hours=food_items[food_item][temp_range])
    print(expire_time)
    response = {
        "city":city,
        "temperature":temperature,
        "expiration_time":  expire_time,
        "is_expired": expire_time < datetime.now(),
        "now" : datetime.now(),
        "label": label
    }
    return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)