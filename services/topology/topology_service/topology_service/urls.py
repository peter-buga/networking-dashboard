from django.urls import path, include

urlpatterns = [
    path('', include('topology_app.urls')),
]
