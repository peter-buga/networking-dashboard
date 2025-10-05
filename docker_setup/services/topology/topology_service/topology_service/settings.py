import os
from pathlib import Path
from django.core.management.utils import get_random_secret_key

DEBUG = os.environ.get('DJANGO_DEBUG', 'False').lower() in ['true', '1', 'yes', 'on']
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1,topology').split(',')
INSTALLED_APPS = [
    'topology_app',
]
ROOT_URLCONF = 'topology_service.urls'
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', get_random_secret_key())
