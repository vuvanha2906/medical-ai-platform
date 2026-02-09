from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from .models import Study
from .serializers import StudyUploadSerializer, StudyDetailSerializer
from apps.ai_engine.services import run_inference_service  # Import service vừa viết


class StudyUploadView(APIView):
    # Hỗ trợ upload file qua Form-data
    parser_classes = (parsers.MultiPartParser, parsers.FormParser)

    def post(self, request, *args, **kwargs):
        serializer = StudyUploadSerializer(data=request.data)

        if serializer.is_valid():
            # 1. Lưu file và tạo record 'PENDING'
            study = serializer.save(status='PROCESSING')

            # 2. Gọi AI Inference
            # Lưu ý: Trong Production, dòng này nên đẩy vào Celery Task (Async)
            # Hiện tại để demo workflow, ta chạy Sync (trực tiếp)
            run_inference_service(study)

            # 3. Trả về kết quả sau khi AI chạy xong
            # Reload lại data từ DB để lấy kết quả mới nhất
            study.refresh_from_db()
            response_serializer = StudyDetailSerializer(study)

            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class StudyRetrieveView(APIView):
    def get(self, request, pk):
        try:
            study = Study.objects.get(pk=pk)
            serializer = StudyDetailSerializer(study)
            return Response(serializer.data)
        except Study.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)