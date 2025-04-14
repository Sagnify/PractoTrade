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

    def __str__(self):
        return f"{self.company_name} - {self.sentiment_category} - {self.sentiment_score}"

    class Meta:
        ordering = ['-timestamp']
