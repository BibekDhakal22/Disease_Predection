from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('change-password/', views.change_password_view, name='change_password'),
    path('about/', views.about_view, name='about'),
    path('stats/', views.stats_view, name='stats'),
    path('history/<int:pk>/export/', views.export_prediction_pdf, name='export_pdf'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
]