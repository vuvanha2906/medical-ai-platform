import os
from django.core.wsgi import get_wsgi_application

# Trỏ đúng vào file base.py trong thư mục cấu hình của bạn
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')

application = get_wsgi_application()