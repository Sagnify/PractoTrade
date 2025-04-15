# task.py
from celery import shared_task
import time

@shared_task
def handle_sleep():
    print("Sleeping for 20 seconds...")
    time.sleep(20)

# @shared_task(bind=True, name='handle_sleep_task')
# def handle_sleep(self):
#     print("Sleeping for 10 seconds...")
#     time.sleep(20)
#     return "Task completed"