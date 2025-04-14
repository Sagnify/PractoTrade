from django.db import models

class CompanySentiment(models.Model):
    company_name = models.CharField(max_length=255)
    sentiment_score = models.FloatField()
    sentiment_category = models.CharField(max_length=50)
    source = models.CharField(max_length=50, choices=[('Reddit', 'Reddit'), ('News', 'News')])
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.company_name} - {self.sentiment_category} - {self.sentiment_score}"

    class Meta:
        ordering = ['-timestamp']
