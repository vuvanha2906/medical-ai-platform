# backend/apps/studies/utils.py
from io import BytesIO
from django.template.loader import render_to_string
from django.http import HttpResponse
from xhtml2pdf import pisa

def render_to_pdf(template_src: str, context_dict: dict = {}) -> HttpResponse:
    """
    Renders an HTML template with context data and returns an HttpResponse
    containing the generated PDF.
    """
    # Render the HTML template with the given context
    html = render_to_string(template_src, context_dict)

    # Create a BytesIO object to hold the PDF
    result = BytesIO()

    # Create the PDF object using xhtml2pdf
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), dest=result, encoding='UTF-8')

    # Check if PDF generation was successful
    if pdf.err:
        return HttpResponse('We had some errors<br />' + pdf.err, status=400)

    # Create the HttpResponse object with the PDF content
    response = HttpResponse(result.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    return response