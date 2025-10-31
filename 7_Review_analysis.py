import pandas as pd
from textblob import TextBlob

# Dummy Data Collection (simulating social media reviews)
data = {
    'review_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'review_text': [
        "This hospital provided excellent care, I'm very satisfied.",
        "The new medication has too many side effects, feeling unwell.",
        "Doctor was professional and the appointment was on time. Neutral experience.",
        "Terrible service, long waiting times and rude staff.",
        "Highly recommend this clinic, friendly staff and effective treatment.",
        "Vaccine caused mild fever, but otherwise fine. A bit concerning.",
        "The staff at the emergency room were incredibly helpful.",
        "Pharmacy was out of stock, very inconvenient.",
        "My recovery has been smooth thanks to the therapy.",
        "Unsure about the treatment effectiveness, need more time to tell."
    ]
}
df = pd.DataFrame(data)

# Preprocessing (simple normalization)
def clean_text(text):
    text = text.lower().strip()
    return text

df['cleaned_review_text'] = df['review_text'].apply(clean_text)

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['cleaned_review_text'].apply(get_sentiment)

# Topic keywords (simplified)
keywords = {
    'side effects': ['side effects', 'unwell', 'fever', 'concerning'],
    'satisfaction': ['excellent care', 'satisfied', 'highly recommend', 'friendly', 'helpful', 'smooth recovery'],
    'service quality': ['professional', 'on time', 'terrible service', 'long waiting times', 'rude staff', 'inconvenient'],
    'treatment effectiveness': ['effective treatment', 'vaccine caused', 'recovery has been smooth', 'unsure about treatment effectiveness']
}

def identify_topics(text):
    found_topics = []
    for topic, kws in keywords.items():
        for kw in kws:
            if kw in text:
                found_topics.append(topic)
                break
    return ", ".join(found_topics) if found_topics else "General"

df['topics'] = df['cleaned_review_text'].apply(identify_topics)

# Visible output
print("Program Output:")
print(df[['review_text', 'sentiment', 'topics']].to_string(index=False))