import json
import joblib
import numpy as np
from pathlib import Path
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.db.models import Count
from .models import Prediction

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / 'ml'

model = joblib.load(ML_DIR / 'model.pkl')
label_encoder = joblib.load(ML_DIR / 'label_encoder.pkl')
SYMPTOMS = joblib.load(ML_DIR / 'symptoms.pkl')

DISEASE_INFO = {
    'Fungal infection':       {'precautions': ['Keep affected area clean', 'Use antifungal cream', 'Avoid sharing personal items', 'Wear breathable clothing'], 'severity': 'Mild'},
    'Allergy':                {'precautions': ['Avoid allergens', 'Take antihistamines', 'Keep windows closed', 'Use air purifier'], 'severity': 'Mild'},
    'GERD':                   {'precautions': ['Avoid spicy food', 'Eat smaller meals', 'Do not lie down after eating', 'Avoid caffeine'], 'severity': 'Moderate'},
    'Chronic cholestasis':    {'precautions': ['Consult a gastroenterologist', 'Low fat diet', 'Avoid alcohol', 'Regular blood tests'], 'severity': 'Severe'},
    'Drug Reaction':          {'precautions': ['Stop medication immediately', 'Consult a doctor', 'Keep record of medications', 'Drink plenty of water'], 'severity': 'Moderate'},
    'Peptic ulcer disease':   {'precautions': ['Avoid NSAIDs', 'Eat regular small meals', 'Avoid spicy food', 'Reduce stress'], 'severity': 'Moderate'},
    'AIDS':                   {'precautions': ['Take antiretroviral therapy', 'Practice safe behaviors', 'Regular medical checkups', 'Maintain healthy diet'], 'severity': 'Severe'},
    'Diabetes':               {'precautions': ['Monitor blood sugar', 'Follow diet plan', 'Exercise regularly', 'Take prescribed medication'], 'severity': 'Moderate'},
    'Gastroenteritis':        {'precautions': ['Drink plenty of fluids', 'Rest', 'Eat bland foods', 'Wash hands frequently'], 'severity': 'Mild'},
    'Bronchial Asthma':       {'precautions': ['Use inhaler as prescribed', 'Avoid triggers', 'Keep rescue inhaler handy', 'Monitor breathing'], 'severity': 'Moderate'},
    'Hypertension':           {'precautions': ['Reduce salt intake', 'Exercise regularly', 'Take medications', 'Monitor blood pressure'], 'severity': 'Moderate'},
    'Migraine':               {'precautions': ['Rest in dark quiet room', 'Stay hydrated', 'Avoid known triggers', 'Take prescribed medication'], 'severity': 'Moderate'},
    'Cervical spondylosis':   {'precautions': ['Neck exercises', 'Avoid straining neck', 'Use ergonomic pillow', 'Hot/cold compress'], 'severity': 'Moderate'},
    'Paralysis (brain hemorrhage)': {'precautions': ['Seek emergency care immediately', 'Physical therapy', 'Follow doctors advice', 'Regular monitoring'], 'severity': 'Severe'},
    'Jaundice':               {'precautions': ['Rest completely', 'Drink plenty of water', 'Avoid fatty foods', 'Consult doctor immediately'], 'severity': 'Moderate'},
    'Malaria':                {'precautions': ['Use mosquito nets', 'Take antimalarial drugs', 'Wear full-sleeve clothes', 'Remove stagnant water'], 'severity': 'Severe'},
    'Chicken pox':            {'precautions': ['Get vaccinated', 'Avoid scratching', 'Use calamine lotion', 'Isolate from others'], 'severity': 'Mild'},
    'Dengue':                 {'precautions': ['Use mosquito repellent', 'Rest and hydrate', 'Monitor platelet count', 'Avoid aspirin'], 'severity': 'Severe'},
    'Typhoid':                {'precautions': ['Drink boiled water', 'Eat freshly cooked food', 'Get vaccinated', 'Maintain hygiene'], 'severity': 'Severe'},
    'Hepatitis A':            {'precautions': ['Get vaccinated', 'Wash hands frequently', 'Avoid contaminated food/water', 'Rest'], 'severity': 'Moderate'},
    'Hepatitis B':            {'precautions': ['Get vaccinated', 'Avoid sharing needles', 'Practice safe behaviors', 'Regular checkups'], 'severity': 'Severe'},
    'Hepatitis C':            {'precautions': ['Avoid sharing needles', 'Consult hepatologist', 'Antiviral treatment', 'Avoid alcohol'], 'severity': 'Severe'},
    'Hepatitis D':            {'precautions': ['Get Hepatitis B vaccine', 'Avoid sharing needles', 'Consult doctor', 'Antiviral medications'], 'severity': 'Severe'},
    'Hepatitis E':            {'precautions': ['Drink clean water', 'Maintain hygiene', 'Avoid raw foods', 'Rest'], 'severity': 'Moderate'},
    'Alcoholic hepatitis':    {'precautions': ['Stop alcohol completely', 'Nutritious diet', 'Consult hepatologist', 'Regular liver tests'], 'severity': 'Severe'},
    'Tuberculosis':           {'precautions': ['Complete medication course', 'Cover mouth when coughing', 'Ventilate living spaces', 'Regular checkups'], 'severity': 'Severe'},
    'Common Cold':            {'precautions': ['Rest', 'Drink warm fluids', 'Use steam inhalation', 'Avoid cold exposure'], 'severity': 'Mild'},
    'Pneumonia':              {'precautions': ['Take prescribed antibiotics', 'Rest and hydrate', 'Deep breathing exercises', 'Get vaccinated'], 'severity': 'Severe'},
    'Dimorphic hemorrhoids':  {'precautions': ['Eat high fiber diet', 'Drink plenty of water', 'Avoid straining', 'Sitz baths'], 'severity': 'Mild'},
    'Heart attack':           {'precautions': ['Seek emergency care immediately', 'Take aspirin if advised', 'CPR if needed', 'Avoid exertion'], 'severity': 'Severe'},
    'Varicose veins':         {'precautions': ['Exercise regularly', 'Elevate legs', 'Wear compression stockings', 'Avoid standing long'], 'severity': 'Mild'},
    'Hypothyroidism':         {'precautions': ['Take thyroid medication', 'Regular blood tests', 'Eat balanced diet', 'Exercise regularly'], 'severity': 'Moderate'},
    'Hyperthyroidism':        {'precautions': ['Take prescribed medication', 'Avoid iodine-rich foods', 'Regular thyroid tests', 'Manage stress'], 'severity': 'Moderate'},
    'Hypoglycemia':           {'precautions': ['Eat regular meals', 'Carry glucose tablets', 'Monitor blood sugar', 'Avoid skipping meals'], 'severity': 'Moderate'},
    'Osteoarthritis':         {'precautions': ['Gentle exercise', 'Maintain healthy weight', 'Use pain relievers as advised', 'Physical therapy'], 'severity': 'Moderate'},
    'Arthritis':              {'precautions': ['Physical therapy', 'Anti-inflammatory medication', 'Warm compresses', 'Gentle exercise'], 'severity': 'Moderate'},
    '(Vertigo) Paroxysmal Positional Vertigo': {'precautions': ['Epley maneuver', 'Move slowly', 'Avoid sudden movements', 'Consult ENT specialist'], 'severity': 'Mild'},
    'Acne':                   {'precautions': ['Wash face gently', 'Avoid touching face', 'Use non-comedogenic products', 'Consult dermatologist'], 'severity': 'Mild'},
    'Urinary tract infection':{'precautions': ['Drink plenty of water', 'Urinate frequently', 'Take antibiotics as prescribed', 'Maintain hygiene'], 'severity': 'Mild'},
    'Psoriasis':              {'precautions': ['Moisturize regularly', 'Avoid triggers', 'Use prescribed topical creams', 'Manage stress'], 'severity': 'Moderate'},
    'Impetigo':               {'precautions': ['Keep wounds clean', 'Use antibiotic cream', 'Avoid touching sores', 'Wash hands frequently'], 'severity': 'Mild'},
}
SEVERITY_COLOR = {'Mild': '#22c55e', 'Moderate': '#f59e0b', 'Severe': '#ef4444'}


# ── Auth Views ────────────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        username = request.POST.get('username')
        email    = request.POST.get('email')
        password = request.POST.get('password')
        confirm  = request.POST.get('confirm')
        if password != confirm:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            login(request, user)
            messages.success(request, f'Welcome, {username}!')
            return redirect('index')
    return render(request, 'predictor/register.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'predictor/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


# ── Main Views ────────────────────────────────────────────────────────────────

@login_required
def index(request):
    symptom_list = sorted([s.replace('_', ' ').title() for s in SYMPTOMS])
    return render(request, 'predictor/index.html', {'symptoms': json.dumps(symptom_list)})


@login_required
def history(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'predictor/history.html', {'predictions': predictions})


@csrf_exempt
@login_required
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    data = json.loads(request.body)
    selected = data.get('symptoms', [])

    if len(selected) < 3:
        return JsonResponse({'error': 'Please select at least 3 symptoms.'}, status=400)

    feature_vec = np.zeros(len(SYMPTOMS))
    for s in selected:
        s_key = s.lower().replace(' ', '_')
        if s_key in SYMPTOMS:
            feature_vec[SYMPTOMS.index(s_key)] = 1

    probs = model.predict_proba([feature_vec])[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    results = []
    for i, idx in enumerate(top3_idx):
        disease = label_encoder.inverse_transform([idx])[0]
        info = DISEASE_INFO.get(disease, {'precautions': ['Consult a doctor'], 'severity': 'Unknown'})
        results.append({
            'disease': disease,
            'confidence': round(float(probs[idx]) * 100, 1),
            'severity': info['severity'],
            'precautions': info['precautions'],
            'severity_color': SEVERITY_COLOR.get(info['severity'], '#94a3b8'),
        })
        # Save top result to database
        if i == 0:
            Prediction.objects.create(
                user=request.user,
                symptoms=', '.join(selected),
                disease=disease,
                confidence=round(float(probs[idx]) * 100, 1),
                severity=info['severity'],
            )

    return JsonResponse({'predictions': results, 'symptoms_used': selected})

@staff_member_required
def admin_dashboard(request):
    from django.contrib.auth.models import User

    total_users       = User.objects.count()
    total_predictions = Prediction.objects.count()
    severe_count      = Prediction.objects.filter(severity='Severe').count()
    mild_count        = Prediction.objects.filter(severity='Mild').count()
    moderate_count    = Prediction.objects.filter(severity='Moderate').count()

    # Top 5 most predicted diseases
    top_diseases = (
        Prediction.objects
        .values('disease')
        .annotate(count=Count('disease'))
        .order_by('-count')[:5]
    )

    # Recent 10 predictions
    recent = Prediction.objects.select_related('user').order_by('-created_at')[:10]

    # All users with prediction count
    users = User.objects.annotate(pred_count=Count('prediction')).order_by('-date_joined')[:10]

    context = {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'severe_count': severe_count,
        'mild_count': mild_count,
        'moderate_count': moderate_count,
        'top_diseases': top_diseases,
        'recent': recent,
        'users': users,
    }
    return render(request, 'predictor/admin_dashboard.html', context)

# ── Password Change ───────────────────────────────────────────────────────────

@login_required
def change_password_view(request):
    if request.method == 'POST':
        current  = request.POST.get('current_password')
        new_pw   = request.POST.get('new_password')
        confirm  = request.POST.get('confirm_password')
        if not request.user.check_password(current):
            messages.error(request, 'Current password is incorrect.')
        elif new_pw != confirm:
            messages.error(request, 'New passwords do not match.')
        elif len(new_pw) < 6:
            messages.error(request, 'Password must be at least 6 characters.')
        else:
            request.user.set_password(new_pw)
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, 'Password changed successfully!')
            return redirect('profile')
    return render(request, 'predictor/change_password.html')


# ── Profile ───────────────────────────────────────────────────────────────────

@login_required
def profile_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '')
        last_name  = request.POST.get('last_name', '')
        email      = request.POST.get('email', '')
        request.user.first_name = first_name
        request.user.last_name  = last_name
        request.user.email      = email
        request.user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('profile')
    pred_count  = Prediction.objects.filter(user=request.user).count()
    severe      = Prediction.objects.filter(user=request.user, severity='Severe').count()
    mild        = Prediction.objects.filter(user=request.user, severity='Mild').count()
    moderate    = Prediction.objects.filter(user=request.user, severity='Moderate').count()
    recent      = Prediction.objects.filter(user=request.user).order_by('-created_at')[:5]
    context = {
        'pred_count': pred_count,
        'severe': severe,
        'mild': mild,
        'moderate': moderate,
        'recent': recent,
    }
    return render(request, 'predictor/profile.html', context)


# ── About ─────────────────────────────────────────────────────────────────────

def about_view(request):
    return render(request, 'predictor/about.html')


# ── Stats ─────────────────────────────────────────────────────────────────────

@login_required
def stats_view(request):
    from django.db.models.functions import TruncDate
    total       = Prediction.objects.count()
    severe      = Prediction.objects.filter(severity='Severe').count()
    mild        = Prediction.objects.filter(severity='Mild').count()
    moderate    = Prediction.objects.filter(severity='Moderate').count()
    top_diseases = (
        Prediction.objects.values('disease')
        .annotate(count=Count('disease'))
        .order_by('-count')[:8]
    )
    # Accuracy data for the 8 most common diseases
    disease_labels = [d['disease'] for d in top_diseases]
    disease_counts = [d['count'] for d in top_diseases]
    context = {
        'total': total,
        'severe': severe,
        'mild': mild,
        'moderate': moderate,
        'top_diseases': top_diseases,
        'disease_labels': json.dumps(disease_labels),
        'disease_counts': json.dumps(disease_counts),
        'accuracy': 99.59,
        'n_diseases': 41,
        'n_symptoms': 132,
        'n_estimators': 200,
    }
    return render(request, 'predictor/stats.html', context)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

@login_required
def export_prediction_pdf(request, pk):
    try:
        prediction = Prediction.objects.get(pk=pk, user=request.user)
    except Prediction.DoesNotExist:
        return redirect('history')

    # Get precautions from DISEASE_INFO
    info = DISEASE_INFO.get(prediction.disease, {
        'precautions': ['Consult a doctor', 'Rest', 'Stay hydrated', 'Monitor symptoms'],
        'severity': prediction.severity
    })

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="MediPredict_{prediction.disease.replace(" ", "_")}_{prediction.created_at.strftime("%Y%m%d")}.pdf"'

    doc = SimpleDocTemplate(response, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    story = []
    styles = getSampleStyleSheet()

    # ── Custom styles ──
    title_style = ParagraphStyle('Title', fontSize=22, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#0f172a'), spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle('Sub', fontSize=10, fontName='Helvetica',
                                textColor=colors.HexColor('#64748b'), spaceAfter=2, alignment=TA_CENTER)
    section_style = ParagraphStyle('Section', fontSize=12, fontName='Helvetica-Bold',
                                    textColor=colors.HexColor('#0f172a'), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('Body', fontSize=10, fontName='Helvetica',
                                 textColor=colors.HexColor('#374151'), spaceAfter=4, leading=16)
    small_style = ParagraphStyle('Small', fontSize=9, fontName='Helvetica',
                                  textColor=colors.HexColor('#94a3b8'), spaceAfter=2)

    SEV_COLOR = {'Mild': '#22c55e', 'Moderate': '#f59e0b', 'Severe': '#ef4444'}
    sev_color = colors.HexColor(SEV_COLOR.get(prediction.severity, '#94a3b8'))

    # ── Header ──
    story.append(Paragraph("🩺 MediPredict", title_style))
    story.append(Paragraph("AI-Powered Disease Prediction System", sub_style))
    story.append(Paragraph("Tribhuvan University — BCA Final Year Project 2021", small_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#e2e8f0')))
    story.append(Spacer(1, 0.4*cm))

    # ── Prediction Report title ──
    story.append(Paragraph("PREDICTION REPORT", ParagraphStyle('PT', fontSize=11,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#6366f1'),
        spaceAfter=12, alignment=TA_CENTER, letterSpacing=2)))

    # ── Info table ──
    info_data = [
        ['Patient / User', prediction.user.username],
        ['Report Date', prediction.created_at.strftime('%B %d, %Y at %I:%M %p')],
        ['Symptoms Entered', prediction.symptoms],
        ['Predicted Disease', prediction.disease],
        ['Confidence Score', f'{prediction.confidence}%'],
        ['Severity Level', prediction.severity],
    ]
    info_table = Table(info_data, colWidths=[4.5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (1,0), (1,-1), colors.white),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#64748b')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#0f172a')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('ROWBACKGROUND', (0,0), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        # Severity color
        ('TEXTCOLOR', (1,5), (1,5), sev_color),
        ('FONTNAME', (1,5), (1,5), 'Helvetica-Bold'),
        # Confidence color
        ('TEXTCOLOR', (1,4), (1,4), colors.HexColor('#6366f1')),
        ('FONTNAME', (1,4), (1,4), 'Helvetica-Bold'),
        # Disease bold
        ('FONTNAME', (1,3), (1,3), 'Helvetica-Bold'),
        ('FONTSIZE', (1,3), (1,3), 11),
        ('TEXTCOLOR', (1,3), (1,3), colors.HexColor('#0f172a')),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Precautions ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0')))
    story.append(Paragraph("Recommended Precautions", section_style))

    prec_data = [[f"{'✓':^3}", p] for p in info['precautions']]
    prec_table = Table(prec_data, colWidths=[0.8*cm, 15.7*cm])
    prec_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#22c55e')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#374151')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUND', (0,0), (-1,-1),
         [colors.HexColor('#f0fdf4'), colors.white]),
        ('LINEBELOW', (0,0), (-1,-2), 0.3, colors.HexColor('#e2e8f0')),
    ]))
    story.append(prec_table)
    story.append(Spacer(1, 0.5*cm))

    # ── What to do next ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0')))
    story.append(Paragraph("What To Do Next", section_style))
    next_steps = [
        "Visit a qualified healthcare provider and share this report with them.",
        "Do not self-medicate based solely on this AI prediction.",
        "If severity is marked as Severe, seek medical attention immediately.",
        "Keep track of your symptoms and note any changes over time.",
        "Follow all precautions listed above to manage your condition.",
    ]
    for step in next_steps:
        story.append(Paragraph(f"• {step}", body_style))

    story.append(Spacer(1, 0.5*cm))

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#fde68a')))
    story.append(Spacer(1, 0.25*cm))
    disclaimer_style = ParagraphStyle('Disc', fontSize=8.5, fontName='Helvetica',
        textColor=colors.HexColor('#92400e'),
        backColor=colors.HexColor('#fffbeb'),
        borderColor=colors.HexColor('#fde68a'),
        borderWidth=1, borderPadding=8,
        spaceAfter=4, leading=14)
    story.append(Paragraph(
        "⚠️ MEDICAL DISCLAIMER: This report is generated by an AI-based educational tool "
        "and is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional for medical concerns. "
        "MediPredict — TU BCA Final Year Project 2021.",
        disclaimer_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Generated by MediPredict on {prediction.created_at.strftime('%B %d, %Y')} | "
        f"User: {prediction.user.username} | Confidence: {prediction.confidence}%",
        ParagraphStyle('Footer', fontSize=8, fontName='Helvetica',
                       textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)))

    doc.build(story)
    return response


@login_required
def chatbot_view(request):
    symptom_list = sorted([s.replace('_', ' ').title() for s in SYMPTOMS])
    return render(request, 'predictor/chatbot.html', {
        'symptoms': json.dumps(symptom_list)
    })