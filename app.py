# from flask import Flask 

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello"

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
#import os
import pickle
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load pre-trained models and data


#BASE_DIR = "E:\image_desc_gen_proj\Image Description Generator"
#WORKING_DIR = "E:\image_desc_gen_proj\Image Description Generator\models"

model = load_model(r"D:\IDG\model_1.h5")




with open(r"D:\IDG\features1.pkl", 'rb') as f:
    features = pickle.load(f)
    
    
#with open(r"Flickr8k.token.txt" 'r') as f:
    #next(f)
 #   captions_doc = f.read()
    

# print(captions_doc)


mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    #print(tokens)
    if(len(line)<2):
        continue
    image_id=tokens[0]
    image_id = image_id.split('.')[0]
    tokens=str(tokens)
    parts = tokens.split(maxsplit=1)
    try:
      caption = parts[1].strip()  # Access description if it exists
    except IndexError:
      caption = ""
    caption = "".join(caption) 
    #print(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption[:-2])


# print(mapping)
# len(mapping)
# print(type(mapping))

#sample before cleaning
# mapping['979383193_0a542a059d']
    
    
#preprocess clean text
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            # delete digits, special chars, etc., 
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # print(caption)
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    
#sample caption output
clean(mapping)
# mapping['979383193_0a542a059d']






all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)
# print(max_length)






vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load tokenizer
# with open(r"E:\image_desc_gen_proj\Image Description Generator\models\tokenizer.pkl", 'rb') as f:
#     tokenizer = pickle.load(f)

# max_length = 34  # Assuming max_length from your preprocessing

# Function to generate caption



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
     if index == integer:
        return word
    return None



# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        # print(sequence)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text



def generate_caption(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the file from the POST request
#         f = request.files['file']
#         # Save the file to ./uploads
#         file_path = "static/uploads/" + f.filename
#         f.save(file_path)
#         # Generate caption
#         # load image
#         image = load_img(file_path, target_size=(224, 224))
#         # convert image pixels to numpy array
#         image = img_to_array(image)
#         # reshape data for model
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         # preprocess image from vgg
#         image = preprocess_input(image)
#         # extract featuresS
#         feature = vgg_model.predict(image, verbose=0)
#         # predict from the trained model
#         output = predict_caption(model,feature,tokenizer,max_length)
#         # caption = generate_caption(file_path)
#         print(output)
#         return render_template('./index.html',image_file=file_path, caption=output)
from flask import jsonify

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        # Save the file to ./uploads
        file_path = "static/uploads/" + f.filename
        f.save(file_path)
        # Generate caption
        # load image
        image = load_img(file_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image from vgg
        image = preprocess_input(image)
        # extract featuresS
        feature = vgg_model.predict(image, verbose=0)
        # predict from the trained model
        output = predict_caption(model, feature, tokenizer, max_length)
        # Return JSON response
        return jsonify({'caption': output})


if __name__ == '__main__':
    app.run(debug=True)
