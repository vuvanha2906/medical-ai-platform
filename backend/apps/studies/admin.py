from django.contrib import admin
from .models import Study

@admin.register(Study)
class StudyAdmin(admin.ModelAdmin):
    list_display = ('id', 'study_type', 'status', 'created_at')
    list_filter = ('status', 'study_type')
    readonly_fields = ('id', 'created_at', 'updated_at', 'ai_results')