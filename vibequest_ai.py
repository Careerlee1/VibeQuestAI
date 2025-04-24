import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import random
import os

# Sample dataset (10 entries to keep it super light)
data = {
    "quiz_responses": [
        "cow tea dance", "robot coffee sing", "cow coffee dance", "robot tea sing",
        "cow tea sing", "robot coffee dance", "cow coffee sing", "robot tea dance",
        "paratha filter reel", "dosa nofilter story"
    ],
    "vibe_profile": [
        "funny traditional", "techy modern", "chill traditional", "artsy modern",
        "playful traditional", "energetic modern", "relaxed traditional", "creative modern",
        "trendy desi", "authentic desi"
    ],
    "challenge": [
        "Take a selfie at a chai stall!", "Make a 10-sec Reel dancing to a song.",
        "Share a photo of street food.", "Sing a line from a song in a Story.",
        "Post a pic with a cow nearby.", "Create a meme about coffee vs. tea.",
        "Share a rain photo with a caption.", "Draw a doodle and post it on X.",
        "Make a Reel at a dosa stall.", "Post a no-filter breakfast Story."
    ],
    "vernacular_prompt": [
        "Gai ya robot? Chai ya coffee? Dance ya sing? (Hindi)",
        "Robot ya gai? Coffee ya chai? Sing ya dance? (Hindi)",
        "Gai ya robot? Coffee ya chai? Dance ya sing? (Hindi)",
        "Robot ya gai? Chai ya coffee? Sing ya dance? (Hindi)",
        "Pashu ya robot? Chai ya coffee? Sing ya dance? (Tamil)",
        "Robot ya pashu? Coffee ya chai? Dance ya sing? (Tamil)",
        "Gai ya robot? Coffee ya chai? Sing ya dance? (Hindi)",
        "Robot ya gai? Chai ya coffee? Dance ya sing? (Hindi)",
        "Paratha ya dosa? Filter ya no-filter? Reel ya Story? (Hindi)",
        "Dosa ya paratha? Filter illaiya? Reel ya Story? (Tamil)"
    ],
    "badge": [
        "Chai Champion", "Dance Star", "Street Foodie", "Song Sparrow",
        "Desi Vibes", "Meme Master", "Rain Guru", "Doodle Pro",
        "Reel Rockstar", "Story Sultan"
    ]
}

# Save dataset to CSV
df = pd.DataFrame(data)
df.to_csv("vibequest_dataset.csv", index=False)

# Load dataset
def load_data():
    return pd.read_csv("vibequest_dataset.csv")

# Train model
def train_model():
    df = load_data()
    X = df["quiz_responses"]
    y = df["vibe_profile"]
    model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=3))
    model.fit(X, y)
    return model, df

# Match vibe and suggest challenge
def match_vibe(user_responses, user_language="English"):
    user_input = " ".join(user_responses).lower().strip()
    model, df = train_model()
    vibe_profile = model.predict([user_input])[0]
    candidates = df[df["vibe_profile"] == vibe_profile]
    if candidates.empty:
        return {"message": "No vibe-match found. Try different answers!"}
    row = candidates.sample(n=1).iloc[0]
    return {
        "vibe": row["vibe_profile"],
        "challenge": row["challenge"],
        "vernacular": row["vernacular_prompt"] if user_language.lower() != "english" else "Save a cow or robot? Tea or coffee? Dance or sing?",
        "badge": row["badge"],
        "message": f"You’re a {row['vibe_profile']} vibe!"
    }

# Main game
def main():
    print("🎉 Welcome to VibeQuest AI! 🎉")
    print("Answer 3 questions to find your vibe and get a fun challenge!")
    print("Q1: Save a cow or robot?")
    print("Q2: Tea or coffee?")
    print("Q3: Dance or sing?")
    while True:
        print("\nType your answers (e.g., 'cow tea dance') or 'quit' to exit:")
        user_input = input("Your answers: ").strip()
        if user_input.lower() == "quit":
            print("Thanks for playing VibeQuest AI! Come back soon! 🎮")
            break
        answers = user_input.split()
        if len(answers) != 3:
            print("Please enter exactly 3 answers (e.g., 'cow tea dance').")
        user_language = input("Language (e.g., 'Tamil', 'Hindi', or Enter for English): ").strip() or "English"
        result = match_vibe(answers, user_language)
        print("\n🎮 " + result["message"])
        if "vibe" in result:
            print(f"Vibe: {result['vibe']}")
            print(f"Challenge: {result['challenge']}")
            print(f"Badge: {result['badge']} 🏆")
            print(f"Questions in {user_language}: {result['vernacular']}")
            print("Share your challenge on Instagram or WhatsApp! 📱")
        print("\nPlay again for a new vibe!")

if __name__ == "__main__":
    main()