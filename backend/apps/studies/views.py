from django.shortcuts import render, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Study, Prediction
from .serializers import StudySerializer
from apps.ai_engine.tasks import process_medical_image
import uuid

class StudyUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = StudySerializer(data=request.data)
        if serializer.is_valid():
            study = serializer.save()
            # Trigger the Celery task
            process_medical_image.delay(str(study.id))
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def DashboardView(request):
    return render(request, 'studies/dashboard.html')

def XrayAnalysisView(request):
    return HttpResponse("X-ray Analysis Module coming soon")

def MriAlzheimerView(request):
    return HttpResponse("MRI Alzheimer Module coming soon")