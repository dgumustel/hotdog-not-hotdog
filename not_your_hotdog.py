
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow
from PIL import Image
import numpy as np



st.title('Not Your Dog')

st.write("So you've got a thing for hotdogs, great! We love them too!")
st.write("Upload your best picture and we'll tell you if it's a hotdog or not.")

@st.cache(ttl=10, max_entries=2)
def analyze_image(file):
    hotdog = Image.open(file)
    # hotdog.save('new_image.jpg')
    rgb_im = hotdog.convert('RGB')
    hotdog_arr = tensorflow.keras.preprocessing.image.img_to_array(rgb_im) / 255
    resized_hotdog = tensorflow.image.resize(hotdog_arr, (256, 256))
    hotdog_array = np.array(resized_hotdog).reshape(1,256,256,3)
# st.write(resized_hotdog)

# run preprocessed image through pickled model
    # model = tensorflow.keras.models.load_model('my_model.hdf5')
    model = tensorflow.keras.models.load_model('my_model2.hdf5')

# with open('', mode='rb') as pickle_in:
#     model = pickle.load(pickle_in)
    results = model.predict(hotdog_array)
    return results


# user upload file
file = st.file_uploader("Upload your image here...", type=["png","jpg"])
# st.write(file)
# preprocess image
if file is not None:
    results = analyze_image(file)
    
    if results[0][0] >= 0.5:
        st.write("What a beautiful hotdog!")
        st.balloons()
    else:
        st.write("I don't know what that is, but it ain't a hotdog...")
    st.image(file, use_column_width=True)

