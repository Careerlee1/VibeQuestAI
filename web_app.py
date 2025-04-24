import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

st.title("🎉 VibeQuest AI 🎉")
st.write("Answer 3 questions to find your vibe and get a fun challenge!")

# Load dataset
df = pd.read_csv("vibequest_dataset.csv")

# Train model
def train_model():
    X = df["quiz_responses"]
    y = df["vibe_profile"]
    model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=3))
    model.fit(X, y)
    return model

# Match vibe
def match_vibe(answers, language):
    user_input = " ".join(answers).lower().strip()
    model = train_model()
    vibe_profile = model.predict([user_input])[0]
    candidates = df[df["vibe_profile"] == vibe_profile]
    if candidates.empty:
        return {"message": "No vibe-match found. Try different answers!"}
    row = candidates.sample(n=1).iloc[0]
    return {
        "vibe": row["vibe_profile"],
        "challenge": row["challenge"],
        "vernacular": row["vernacular_prompt"] if language.lower() != "english" else "Save a cow or robot? Tea or coffee? Dance or sing?",
        "badge": row["badge"],
        "message": f"You’re a {row['vibe_profile']} vibe!"
    }

# Web interface
q1 = st.selectbox("Q1: Save a cow or robot?", ["Cow", "Robot"])
q2 = st.selectbox("Q2: Tea or coffee?", ["Tea", "Coffee"])
q3 = st.selectbox("Q3: Dance or sing?", ["Dance", "Sing"])
language = st.selectbox("Language", ["English", "Hindi", "Tamil"])

if st.button("Find My Vibe!"):
    answers = [q1.lower(), q2.lower(), q3.lower()]
    result = match_vibe(answers, language)
    st.write(f"🎮 {result['message']}")
    if "vibe" in result:
        st.write(f"**Vibe**: {result['vibe']}")
        st.write(f"**Challenge**: {result['challenge']}")
        st.write(f"**Badge**: {result['badge']} 🏆")
        st.write(f"**Questions in {language}**: {result['vernacular']}")
        st.write("Share on Instagram or WhatsApp! 📱")