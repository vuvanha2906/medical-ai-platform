from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from .models import Study
from .serializers import StudyUploadSerializer, StudyDetailSerializer
from apps.ai_engine.services import run_inference_service  # Import service vừa viết
from django.shortcuts import render, redirect
from django.views import View
from .forms import StudyForm
from apps.ai_engine.services import run_inference_service

# View xử lý upload (Thay thế API Upload nếu dùng Form thường)
class StudyUploadFormView(View):
    def post(self, request):
        study_type = request.POST.get('study_type')
        image = request.FILES.get('image')

        # Tạo record
        study = Study.objects.create(study_type=study_type, image=image, status='PROCESSING')

        # Gọi AI (Sync hoặc Async)
        # run_inference_service(study)

        # Redirect sang trang kết quả
        return redirect('study-result-html', pk=study.id)


def home_view(request):
    """Render trang chủ (Dashboard)"""
    return render(request, 'home.html')


def upload_study_view(request):
    """Xử lý submit form từ trang chủ"""
    if request.method == 'POST':
        form = StudyForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Lưu record vào DB nhưng chưa commit (để sửa status trước)
            study = form.save(commit=False)
            study.status = 'PROCESSING'
            study.save()

            # 2. Gọi AI Engine (Chạy đồng bộ để demo ngay kết quả)
            # Lưu ý: Trong thực tế, bước này nên đẩy vào Celery Worker
            try:
                run_inference_service(study)
            except Exception as e:
                print(f"AI Error: {e}")

            # 3. Chuyển hướng sang trang kết quả
            return redirect('study-result', pk=study.id)

    # Nếu form lỗi hoặc method GET, quay về trang chủ
    return redirect('home-view')


def result_view(request, pk):
    """Render trang kết quả chẩn đoán"""
    study = get_object_or_404(Study, pk=pk)
    return render(request, 'result.html', {'study': study})

class StudyRetrieveView(APIView):
    def get(self, request, pk):
        try:
            study = Study.objects.get(pk=pk)
            serializer = StudyDetailSerializer(study)
            return Response(serializer.data)
        except Study.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

