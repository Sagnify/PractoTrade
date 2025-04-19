import os
from celery import Celery

# Set default Django settings for 'celery'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksentiment.settings')
app = Celery('stocksentiment')
# Load config from Django settings, namespace 'CELERY'
app.config_from_object('django.conf:settings', namespace='CELERY')
# Autodiscover tasks from all installed apps
app.autodiscover_tasks()



@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')



from datetime import timedelta

app.conf.beat_schedule = {
    'analyze-all-every-20-seconds': {
        'task': 'core.tasks.company_wise_sentiment_analysis',
        'schedule': timedelta(minutes=5),
        'args': (),  # Explicitly no arguments

    },
}


