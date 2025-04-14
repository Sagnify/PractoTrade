from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

text = "TCS stock is performing exceptionally well today 🚀"
score = analyzer.polarity_scores(text)

print("Compound Score:", score["compound"])
print("Breakdown:", score)
