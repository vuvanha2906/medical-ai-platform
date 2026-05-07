# backend/config/celery.py
import os
from celery import Celery

# Set the default Django settings module (change 'base' to 'local' if you run a different file)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')

app = Celery('config')

# namespace='CELERY' means all celery-related configuration keys
# should have a `CELERY_` prefix in your settings.py.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()