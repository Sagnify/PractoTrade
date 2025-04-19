import uuid
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
    predicted_price_with_arima = models.FloatField()
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



class DailyPoll(models.Model):
    company_name = models.CharField(max_length=50)  # Use company name or ticker directly
    question = models.CharField(max_length=255)
    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"Poll for {self.company_name} on {self.created_at}"


class PollOption(models.Model):
    poll = models.ForeignKey(DailyPoll, related_name="options", on_delete=models.CASCADE)
    option_text = models.CharField(max_length=100)
    votes = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.option_text} ({self.votes} votes)"


class Vote(models.Model):
    poll = models.ForeignKey(DailyPoll, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=255)  # Anonymous session/user from localStorage
    option = models.ForeignKey(PollOption, on_delete=models.CASCADE)
    voted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('poll', 'session_id')  # Enforce 1 vote per poll per session

    def __str__(self):
        return f"Vote on {self.poll} by {self.session_id}"
    


from django.contrib.auth.models import User

class Viewer(models.Model):

    username = models.OneToOneField(User, on_delete=models.CASCADE, max_length=20)
    viewer_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)  # You'll have to hash manually

    def __str__(self):
        return self.username

class favourite(models.Model):
    user = models.ForeignKey(Viewer, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.user.username} - {self.company_name}"


