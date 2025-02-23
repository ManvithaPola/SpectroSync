import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

# Define genres dictionary with playable MP3 and video URLs
genres = {
    "Blues": [
        {"name": "The Thrill Is Gone - B.B. King", "audio": "https://www.bensound.com/bensound-music/bensound-slowmotion.mp3", "video": "https://www.youtube.com/watch?v=4fk2prKnYnI"},
        {"name": "Sweet Home Chicago - Robert Johnson", "audio": "https://www.bensound.com/bensound-music/bensound-moose.mp3", "video": "https://www.youtube.com/watch?v=O8hqGu-leFc"},
        {"name": "Stormy Monday - T-Bone Walker", "audio": "https://www.bensound.com/bensound-music/bensound-sunny.mp3", "video": "https://www.youtube.com/watch?v=V-OYKd8SVrI"}
    ],
    "Pop": [
        {"name": "Thriller - Michael Jackson", "audio": "https://www.bensound.com/bensound-music/bensound-dreams.mp3", "video": "https://www.youtube.com/watch?v=sOnqjkJTMaA"},
        {"name": "Shape of You - Ed Sheeran", "audio": "https://www.bensound.com/bensound-music/bensound-summer.mp3", "video": "https://www.youtube.com/watch?v=JGwWNGJdvx8"},
        {"name": "Blinding Lights - The Weeknd", "audio": "https://www.bensound.com/bensound-music/bensound-funkyelement.mp3", "video": "https://www.youtube.com/watch?v=4NRXx6U8ABQ"}
    ],
    "Rock": [
        {"name": "Bohemian Rhapsody - Queen", "audio": "https://www.bensound.com/bensound-music/bensound-energy.mp3", "video": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"},
        {"name": "Stairway to Heaven - Led Zeppelin", "audio": "https://www.bensound.com/bensound-music/bensound-actionable.mp3", "video": "https://www.youtube.com/watch?v=QkF3oxziUI4"},
        {"name": "Hotel California - Eagles", "audio": "https://www.bensound.com/bensound-music/bensound-buddy.mp3", "video": "https://www.youtube.com/watch?v=09839DpTctY"}
    ],
    "Hip-Hop": [
        {"name": "Lose Yourself - Eminem", "audio": "https://www.bensound.com/bensound-music/bensound-hipjazz.mp3", "video": "https://www.youtube.com/watch?v=_Yhyp-_hX2s"},
        {"name": "Juicy - The Notorious B.I.G.", "audio": "https://www.bensound.com/bensound-music/bensound-thejazzpiano.mp3", "video": "https://www.youtube.com/watch?v=_JZom_gVfuw"},
        {"name": "Sicko Mode - Travis Scott", "audio": "https://www.bensound.com/bensound-music/bensound-brazilsamba.mp3", "video": "https://www.youtube.com/watch?v=6ONRf7h3Mdk"}
    ],
    "Jazz": [
        {"name": "So What - Miles Davis", "audio": "https://www.bensound.com/bensound-music/bensound-jazzyfrenchy.mp3", "video": "https://www.youtube.com/watch?v=zqNTltOGh5c"},
        {"name": "Take Five - Dave Brubeck", "audio": "https://www.bensound.com/bensound-music/bensound-theelevatorbossanova.mp3", "video": "https://www.youtube.com/watch?v=vmDDOFXSgAs"},
        {"name": "Feeling Good - Nina Simone", "audio": "https://www.bensound.com/bensound-music/bensound-tenderness.mp3", "video": "https://www.youtube.com/watch?v=OfJRX-8SXOs"}
    ],
    "Disco": [
        {"name": "Stayin' Alive - Bee Gees", "audio": "", "video": "https://www.youtube.com/watch?v=I_izvAbhExY"},
        {"name": "Le Freak - Chic", "audio": "", "video": "https://www.youtube.com/watch?v=h1qQ1SKNlgY"},
        {"name": "Funky Town - Lipps Inc.", "audio": "", "video": "https://www.youtube.com/watch?v=kcKTGZVITD4"}
    ],
    "Metal": [
        {"name": "Master of Puppets - Metallica", "audio": "", "video": "https://www.youtube.com/watch?v=E0ozmU9cJDg"},
        {"name": "Painkiller - Judas Priest", "audio": "", "video": "https://www.youtube.com/watch?v=nM__lPTWThU"},
        {"name": "Holy Wars - Megadeth", "audio": "", "video": "https://www.youtube.com/watch?v=9d4ui9q7eDM"}
    ],
    "Classical": [
        {"name": "Symphony No. 5 - Beethoven", "audio": "https://www.bensound.com/bensound-music/bensound-november.mp3", "video": "https://www.youtube.com/watch?v=fOk8Tm815lE"},
        {"name": "Canon in D - Pachelbel", "audio": "https://www.bensound.com/bensound-music/bensound-littleplanet.mp3", "video": "https://www.youtube.com/watch?v=NlprozGcs80"},
        {"name": "Clair de Lune - Debussy", "audio": "https://www.bensound.com/bensound-music/bensound-tenderness.mp3", "video": "https://www.youtube.com/watch?v=ea2WoUtbzuw"}
    ],
    "Reggae": [
        {"name": "No Woman, No Cry - Bob Marley", "audio": "https://www.bensound.com/bensound-music/bensound-acousticguitar.mp3", "video": "https://www.youtube.com/watch?v=IT8XvzIfiZs"},
        {"name": "Bad Boys - Inner Circle", "audio": "https://www.bensound.com/bensound-music/bensound-cool.mp3", "video": "https://www.youtube.com/watch?v=Do0YXQ51pXI"},
        {"name": "Sweat (A La La La La Long) - Inner Circle", "audio": "https://www.bensound.com/bensound-music/bensound-happyrock.mp3", "video": "https://www.youtube.com/watch?v=7ZfYgGTAGp0"}
    ],
    "Country": [
        {"name": "Take Me Home, Country Roads - John Denver", "audio": "https://www.bensound.com/bensound-music/bensound-riverside.mp3", "video": "https://www.youtube.com/watch?v=1vrEljMfXYo"},
        {"name": "Jolene - Dolly Parton", "audio": "https://www.bensound.com/bensound-music/bensound-easygoing.mp3", "video": "https://www.youtube.com/watch?v=Mc3INQ4ndG8"},
        {"name": "The Gambler - Kenny Rogers", "audio": "https://www.bensound.com/bensound-music/bensound-sunny.mp3", "video": "https://www.youtube.com/watch?v=2I9AOSZxhb0"}
    ]
}

# Define Spotify-inspired color scheme and enhanced UI
st.markdown(
    """
    <style>
     .stApp {
        background-image: url('https://your-background-url.com/bg.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stSidebar, .css-1d391kg, .css-18e3th9 {
        background-color: #FFFFFF !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .stButton>button {
        background-color: #1DB954 !important;
        color: white;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
        padding: 10px 30px;
        border: none;
        cursor: pointer;
    }
    .stButton>button {
        background-color: #1DB954 !important;
        color: white;
        border-radius: 50px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        padding: 10px 30px;
        border: none;
        cursor: pointer;
    }
     .stButton>button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.1);
    }
    .stTextInput>div>div>input {
        color: white;
        background-color: #181818;
        border-radius: 5px;
    }
    .stAudio {
        background-color: #181818 !important;
        border-radius: 10px;
    }
    .stSelectbox>div>div>select {
        color: white;
        background-color: #181818;
        border-radius: 5px;
    }
     .card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .card:hover {
        transform: translateY(-10px);
    }
    .card h4 {
        color: white;
    }
    .card p {
        color: #B3B3B3;
    }
    .stFileUploader>div>div>div {
        background-color: #181818 !important;
        border-radius: 10px;
        color: white;
    }
    .css-1d391kg {
        color: white;
        font-size: 28px;
    }

    .css-1f6a9gk {
        background-color: black; /* Dark grey for sidebar background */
        color: white;
    }

    # <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    # <h1 style='text-align: center; font-size: 50px; color:white;'>
    #     <i class="fa-solid fa-music"></i>
    # </h1>
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# TensorFlow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Sidebar Navigation
st.sidebar.title("🎵 Music Genre Classifier")
app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "🎼 Genre Library", "🎤 Predict"])


# Home Page
if app_mode == "🏠 Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://img.freepik.com/free-photo/abstract-uv-ultraviolet-light-composition_23-2149243965.jpg?t=st=1740317568~exp=1740321168~hmac=15c6550b8abc69f180ce9957cb703ad6c98eb845a411fd9709772ca71e9bf206&w=1800');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            height: 100vh;
        }

        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.4);  /* Added overlay with reduced opacity */
            z-index: -1;
        }

        h1, h2, h3, h4, h5, h6 {
            color: white;  /* Set all headers to white */
            font-family: 'Arial', sans-serif;
            text-align: center;
        }

        /* Music Genre Classification System title in red */
        h1 {
            color: white;  /* Set only the main title to red */
        }

        /* Hover effect for the title */
        h3:hover {
            color: #1DB954;
            transform: scale(1.1);
            transition: all 0.3s ease;
        }

        /* Genre list styling */
        ul li {
            font-size: 28px;
            color: white;  /* Changed text color to white */
            margin: 10px 0;
            transition: all 0.3s ease;
        }

        ul li:hover {
            color: #1DB954;  /* Spotify green on hover */
            transform: translateX(10px);
        }

        .stMarkdown {
            color: white;  /* Apply white color to other texts */
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # Set title in red
    st.markdown("##     Music Genre Classification System 🎷")
    st.markdown("## Discover Music from Various Genres")

    # Genre list and other text
    st.markdown(
        """
        <h3 style="font-size: 36px; color: white;">Explore different music genres:</h3>
        <ul style="font-size:20px;">
            <li>🎸 Rock</li>
            <li>🎻 Classical</li>
            <li>🎷 Jazz</li>
            <li>🎤 HipHop</li>
            <li>🎼 Blues</li>
            <li>💃 Disco</li>
            <li>🎵 Pop</li>
            <li>🤘 Metal</li>
            <li>🌴 Reggae</li>
            <li>🎶 Country</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    # Bottom description in white
    st.markdown(
        """
        Click on *Genre Library* to browse songs from each genre or go to *Predict* to classify an uploaded audio file!
        """,
        unsafe_allow_html=True
    )
# Genre Library Page
elif app_mode == "🎼 Genre Library":
    st.title("🎵 Explore Music by Genre")
    selected_genre = st.selectbox("Select a Genre", list(genres.keys()))
    st.markdown(f"### {selected_genre} Songs", unsafe_allow_html=True)
    
    st.write(f"Here are some popular {selected_genre} songs:")
    for song in genres[selected_genre]:
        st.write(f"🎶 {song['name']}")
        choice = st.radio("Choose media type", ["Audio", "Video"], key=song['name'])
        if choice == "Audio":
            st.audio(song['audio'])
        elif choice == "Video":
            st.video(song['video'])

# Prediction Page
elif app_mode == "🎤 Predict":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3 is not None:
        filepath = 'Test_Music/' + test_mp3.name
        st.write(f"File uploaded: {test_mp3.name}")

        # Play Audio Button
        if st.button("Play Audio"):
            st.audio(test_mp3)

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                try:
                    X_test = load_and_preprocess_data(filepath)
                    st.write(f"Preprocessed Data Shape: {X_test.shape}")  # Debugging step to check the shape
                    result_index = model_prediction(X_test)
                    st.balloons()
                    label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                    st.markdown(f":blue[Model Prediction:] It's a :red[{label[result_index]}] music**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")