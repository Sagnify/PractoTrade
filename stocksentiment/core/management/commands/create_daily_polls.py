from django.core.management.base import BaseCommand
from core.models import DailyPoll, PollOption
import random
from datetime import date

QUESTION_TEMPLATES = [
    "How do you feel about {company} stock for tomorrow?",
    "What's your sentiment on {company} for the next trading day?",
    "Will {company} stock rise or fall tomorrow?",
    "Your opinion: {company} stock's performance tomorrow?",
]

OPTION_CHOICES = ['Bullish', 'Bearish', 'Neutral']

COMPANY_LIST = [
    'META', 'TSLA', 'MSFT', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'WIPRO.NS',
    'HINDUNILVR.NS', 'AMZN', 'GOOGL', 'NVDA', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS'
]

class Command(BaseCommand):
    help = 'Creates daily polls for each company using question templates'

    def handle(self, *args, **kwargs):
        today = date.today()
        for ticker in COMPANY_LIST:
            # Check if a poll already exists for this company today
            if not DailyPoll.objects.filter(company_name=ticker, created_at=today).exists():
                question = random.choice(QUESTION_TEMPLATES).format(company=ticker)
                # Create a new poll for the company
                poll = DailyPoll.objects.create(company_name=ticker, question=question)
                # Create poll options (Bullish, Bearish, Neutral)
                for option in OPTION_CHOICES:
                    PollOption.objects.create(poll=poll, option_text=option)
                self.stdout.write(self.style.SUCCESS(f'Created poll for {ticker}'))
            else:
                self.stdout.write(f'Poll already exists for {ticker}')

