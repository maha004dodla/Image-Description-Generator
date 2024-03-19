import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
TF_ENABLE_ONEDNN_OPTS=0
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


BASE_DIR = "E:\base_dir"
WORKING_DIR = 'E:/base_dir/working_dir'



batch_size = 64
size = (256, 256)
num_channels = 3



model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # restructure the model
print(model.summary()) # summarize



features = {}
directory = os.path.join(r'E:\base_dir\Flickr8k_Dataset\Flicker8k_Dataset')
#print(os.listdir(directory)[:])
for img_name in tqdm(os.listdir(directory)):
    #print(img_name)
    img_path = directory + '/' + img_name # load the image from fil
    #print(img_path)
    image = load_img(img_path, target_size=(224,224))
    image = img_to_array(image) # convert image pixels to numpy array
    #print(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) #resize the image
    image = preprocess_input(image) # preprocess image for vgg
    feature = model.predict(image, verbose=0) # extract features
    image_id = img_name.split('.')[0] # get image ID
    features[image_id] = feature # store feature
    
print(features)


# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR,'features.pkl'), 'wb'))

    
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR,r'E:/base_dir/Flickr8k_text/Flickr8k.token.txt'), 'r') as f:
    #next(f)
    captions_doc = f.read()
    

print(captions_doc)


# create mapping of image to captions
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
    
    
    
    


print(mapping)
len(mapping)
print(type(mapping))

#sample before cleaning
mapping['979383193_0a542a059d']
mapping['1001773457_577c3a7d70']
    
#preprocess clean text
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            print(caption)
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    
#sample caption output
clean(mapping)
mapping['979383193_0a542a059d']


#text preprocessing
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
        
        
len(all_captions)
all_captions[:10]



# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)



# get maximum length of the caption available

max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)


#spliting training and testing dataset images
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


print(train)
print(test)
    


photos=mapping.keys()
descriptions=mapping.values()


print(type(mapping))

from numpy import array
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


def data_generator(mapping, photos, tokenizer, max_length,vocab_size):
    while True:
        for key, desc_list in mapping.items():
            # retrieve the photo feature
            if key in photos:
               photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield (in_img, in_seq), out_word

print(features)

# feature extractor model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
# tie it together [image, seq] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
    
# summarize model
print(model.summary())


from keras.models import model_from_json
from keras.models import load_model
epochs = 15
steps = len(train)
for i in range(epochs):
    generator = data_generator(mapping,features, tokenizer, max_length, vocab_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save('IDGmodel_' + str(i) + '.h5')
    print("Saved model to disk")

    
# save the model
model.save(WORKING_DIR+'/best_IDGmodel.h5')

json_file = open("model.json",'r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r"E:\base_dir\working_dir\best_IDGmodel.h5")
print("Loaded model from disk") 



'''def extract_features(filename, model):
        try:
            image = Image.open(filename) 
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
     if index == integer:
        return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)# predict next word
        yhat = np.argmax(yhat)# get index with high probability
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word 
        if word == 'endseq':
            break
    return in_text[8:-6]





from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    captions = mapping[key]# get actual caption
    #print(captions)
    #print(key,type(key))
    y_pred = predict_caption(model, features[key], tokenizer, max_length)# predict the caption for image
    actual_captions = [caption.split() for caption in captions]# split into words
    y_pred = y_pred.split()
    actual.append(actual_captions)# append to the list
    predicted.append(y_pred)
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))



#print(features)
#print(os.listdir(directory))


from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR,r"E:\base_dir\Flickr8k_Dataset\Flicker8k_Dataset", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    
    
      
generate_caption("1007129816_e794419615.jpg")
generate_caption("1003163366_44323f5815.jpg")
generate_caption("1001773457_577c3a7d70.jpg")


vgg_model = VGG16() 
vgg_model = Model(inputs=vgg_model.inputs,outputs=vgg_model.layers[-2].output)




def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


image_path = r"C:\Users\sirij\Downloads\3.jfif"
image = load_img(image_path, target_size=(224, 224))# load image
image = img_to_array(image)# convert image pixels to numpy array
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
feature = vgg_model.predict(image, verbose=0)# extract features
predict_caption(model, feature, tokenizer, max_length)# predict from the trained model




photo = extract_features(img_path,model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)'''


from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.text 
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import Model
import tensorflow

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


img_path = r"C:\Users\sirij\Downloads\2.jfif"
max_length = 32
'''xception_model = Xception(include_top=False, pooling="avg")
tokenizer = load(open(r"E:\base_dir\working_dir\tokenizer.p","rb"))
model = load_model(r"E:\base_dir\working_dir\model_9.h5")
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)'''



photo = extract_features(img_path, loaded_model)
img = Image.open(img_path)

description = generate_desc(loaded_model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)
