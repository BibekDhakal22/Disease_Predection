from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import Prediction
class CustomAdminSite(admin.AdminSite):
    class Media:
        css = {
            'all': ('admin/css/custom_admin.css',)
        }

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display  = ('user', 'disease', 'severity', 'confidence', 'symptoms_short', 'created_at')
    list_filter   = ('severity', 'disease', 'created_at')
    search_fields = ('user__username', 'disease', 'symptoms')
    ordering      = ('-created_at',)
    readonly_fields = ('user', 'disease', 'symptoms', 'confidence', 'severity', 'created_at')

    def symptoms_short(self, obj):
        return obj.symptoms[:60] + '...' if len(obj.symptoms) > 60 else obj.symptoms
    symptoms_short.short_description = 'Symptoms'


# Customize User display
admin.site.unregister(User)

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'date_joined', 'is_staff', 'prediction_count')
    ordering = ('-date_joined',)

    def prediction_count(self, obj):
        return Prediction.objects.filter(user=obj).count()
    prediction_count.short_description = 'Predictions'


# Customize admin site header
admin.site.site_header  = 'MediPredict Admin'
admin.site.site_title   = 'MediPredict'
admin.site.index_title  = 'Dashboard'