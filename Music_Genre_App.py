import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

# Function to load the model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Model Prediction Function
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Welcome to the,\n## Music Genre Classification System! 🎶🎧")
    
    # Image Handling with Error Handling
    image_path = "music_genre_home.png"
    try:
        st.image(image_path, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image file `music_genre_home.png` not found. Please check your file path.")

    st.markdown("""
    **Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

    ### How It Works
    1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
    2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
    3. **Results:** View the predicted genre along with related information.

    ### Why Choose Us?
    - **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
    - **User-Friendly:** Simple and intuitive interface for a smooth user experience.
    - **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

    ### Get Started
    Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

    ### About Us
    Learn more about the project, our team, and our mission on the **About** page.
    """)

# About Project Page
elif app_mode == "About Project":
    st.markdown("""
    ### About Project
    Music experts have been trying for a long time to understand sound and what differentiates one song from another. How to visualize sound. What makes a tone different from another.

    This data hopefully can give the opportunity to do just that.

    ### About Dataset
    #### Content
    1. **Genres Original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
    2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    3. **Images Original** - A visual representation for each audio file. The audio files were converted to Mel Spectrograms to enable classification using CNN models.
    4. **CSV Files** - Two CSV files contain extracted features from the audio files. One file contains features for full 30-second clips, while the other contains features for split 3-second audio clips (increasing data volume for better classification performance).
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3 is not None:
        filepath = 'Test_Music/' + test_mp3.name

    # Play Audio Button
    if st.button("Play Audio"):
        st.audio(test_mp3)

    # Predict Button
    if st.button("Predict"):
        with st.spinner("Please Wait.."):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            st.balloons()
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.markdown(f"**:blue[Model Prediction:] It's a :red[{label[result_index]}] music**")
