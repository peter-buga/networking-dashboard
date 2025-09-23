from django.urls import path
from . import views

urlpatterns = [
    path('topologyImage', views.topology_image, name='topology_image'),
    path('topologyJson', views.topology_json, name='topology_json'),
    path('health', views.health, name='health'),
]
