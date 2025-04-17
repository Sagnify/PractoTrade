from django.db import models

class CompanySentiment(models.Model):
    company_name = models.CharField(max_length=255)
    sentiment_score = models.FloatField()
    reddit_score = models.FloatField(null=True, blank=True)
    news_score = models.FloatField(null=True, blank=True)
    sentiment_category = models.CharField(max_length=50, choices=[
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral')
    ])
    timestamp = models.DateTimeField(auto_now_add=True)
    stock_data = models.JSONField()


    def __str__(self):
        return f"{self.company_name} - {self.sentiment_category} - {self.sentiment_score}"

    class Meta:
        ordering = ['-timestamp']

class StockPrediction(models.Model):
    company_name = models.CharField(max_length=100)
    predicted_price_with_sentiment = models.FloatField()
    predicted_price_without_sentiment = models.FloatField()
    avg_predicted_price = models.FloatField()
    prediction_time = models.DateTimeField()
    predicted_percentage_change = models.FloatField()
    direction = models.CharField(max_length=10, choices=[
        ('up', 'Up'),
        ('down', 'Down')
    ])
    # sentiment_score = models.FloatField()
    # prediction_interval = models.CharField(max_length=20)  # e.g., "Next 6 Hours"

    def __str__(self):
        return f"{self.company_name} Prediction @ {self.prediction_time}"
