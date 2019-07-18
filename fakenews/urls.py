from django.urls import path
from . import views

urlpatterns=[
    path('', views.home,name="fake-home"),    
    path('result/', views.result),
]