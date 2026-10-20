import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)

# Read the CSV file
df = pd.read_csv("SIT_Canteen_Google_Form_Feedback.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Initialize VADER
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(feedback):
    if pd.isna(feedback) or str(feedback).strip() == "":
        return pd.Series([0.0, "Neutral"])

    score = sia.polarity_scores(str(feedback))["compound"]

    if score >= 0.05:
        sentiment = "Positive"
    elif score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series([score, sentiment])


# Use the actual column name from your CSV
df[["Sentiment_Score", "Sentiment"]] = (
    df["Feedback_Text"].apply(analyze_sentiment)
)

# Display the results
print(df[[
    "Feedback_ID",
    "Feedback_Text",
    "Sentiment_Score",
    "Sentiment"
]].to_string(index=False))

# Save results
df.to_csv("Canteen_Feedback_with_Sentiment.csv", index=False)

print("\nSentiment analysis completed successfully!")
print("Output: Canteen_Feedback_with_Sentiment.csv")

sentiment_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
)

# Define pie chart colors
colors = ["#4CAF50", "#F44336", "#FFC107"]

# Create pie chart
plt.figure(figsize=(8, 8))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    explode=(0.05, 0.05, 0.05),
    shadow=True,
    textprops={"fontsize": 12}
)

plt.title("Canteen Feedback Sentiment Analysis", fontsize=16)
plt.axis("equal")
plt.tight_layout()

# Save the pie chart as an image
plt.savefig("Canteen_Sentiment_Pie_Chart.png", dpi=300)

# Display the chart
plt.show()