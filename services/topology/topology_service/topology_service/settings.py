import os
from pathlib import Path

DEBUG = True
ALLOWED_HOSTS = ['*']
INSTALLED_APPS = [
    'topology_app',
]
ROOT_URLCONF = 'topology_service.urls'
SECRET_KEY = 'your-secret-key'
