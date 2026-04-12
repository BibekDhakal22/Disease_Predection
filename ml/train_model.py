import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
    'vomiting','burning_micturition','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
    'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of_urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
    'belly_pain','abnormal_menstruation','dischromic_patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum',
    'lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples',
    'blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

DISEASE_SYMPTOM_MAP = {
    'Fungal infection':       ['itching','skin_rash','nodal_skin_eruptions','dischromic_patches'],
    'Allergy':                ['continuous_sneezing','shivering','chills','watering_from_eyes'],
    'GERD':                   ['stomach_pain','acidity','ulcers_on_tongue','vomiting','cough','chest_pain'],
    'Chronic cholestasis':    ['itching','vomiting','yellowish_skin','nausea','loss_of_appetite','abdominal_pain','yellowing_of_eyes'],
    'Drug Reaction':          ['itching','skin_rash','stomach_pain','burning_micturition','fatigue'],
    'Peptic ulcer disease':   ['vomiting','indigestion','loss_of_appetite','abdominal_pain','passage_of_gases','internal_itching'],
    'AIDS':                   ['muscle_wasting','patches_in_throat','high_fever','extra_marital_contacts','fatigue'],
    'Diabetes':               ['fatigue','weight_loss','restlessness','lethargy','irregular_sugar_level','polyuria','increased_appetite','excessive_hunger'],
    'Gastroenteritis':        ['vomiting','sunken_eyes','dehydration','diarrhoea'],
    'Bronchial Asthma':       ['fatigue','cough','high_fever','breathlessness','family_history','mucoid_sputum'],
    'Hypertension':           ['headache','chest_pain','dizziness','loss_of_balance','lack_of_concentration'],
    'Migraine':               ['acidity','indigestion','headache','blurred_and_distorted_vision','excessive_hunger','stiff_neck','depression','irritability','visual_disturbances'],
    'Cervical spondylosis':   ['back_pain','weakness_in_limbs','neck_pain','dizziness','loss_of_balance'],
    'Paralysis (brain hemorrhage)': ['vomiting','headache','weakness_of_one_body_side','altered_sensorium','slurred_speech'],
    'Jaundice':               ['itching','vomiting','fatigue','weight_loss','high_fever','yellowish_skin','dark_urine','abdominal_pain'],
    'Malaria':                ['chills','vomiting','high_fever','sweating','headache','nausea','diarrhoea','muscle_pain'],
    'Chicken pox':            ['itching','skin_rash','fatigue','lethargy','high_fever','headache','loss_of_appetite','mild_fever','swelled_lymph_nodes','malaise','red_spots_over_body'],
    'Dengue':                 ['skin_rash','chills','joint_pain','vomiting','fatigue','high_fever','headache','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','malaise','muscle_pain','red_spots_over_body'],
    'Typhoid':                ['chills','vomiting','fatigue','high_fever','headache','nausea','constipation','abdominal_pain','diarrhoea','toxic_look_(typhos)','belly_pain'],
    'Hepatitis A':            ['joint_pain','vomiting','yellowish_skin','dark_urine','nausea','loss_of_appetite','abdominal_pain','diarrhoea','mild_fever','yellowing_of_eyes','muscle_pain'],
    'Hepatitis B':            ['itching','fatigue','lethargy','yellowish_skin','dark_urine','loss_of_appetite','abdominal_pain','yellow_urine','yellowing_of_eyes','malaise','receiving_blood_transfusion','receiving_unsterile_injections'],
    'Hepatitis C':            ['fatigue','yellowish_skin','nausea','loss_of_appetite','yellowing_of_eyes','family_history'],
    'Hepatitis D':            ['joint_pain','vomiting','fatigue','yellowish_skin','dark_urine','nausea','loss_of_appetite','abdominal_pain','yellowing_of_eyes'],
    'Hepatitis E':            ['joint_pain','vomiting','fatigue','high_fever','yellowish_skin','dark_urine','nausea','loss_of_appetite','abdominal_pain','acute_liver_failure','coma','stomach_bleeding'],
    'Alcoholic hepatitis':    ['vomiting','yellowish_skin','abdominal_pain','swelling_of_stomach','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload'],
    'Tuberculosis':           ['chills','vomiting','fatigue','weight_loss','cough','high_fever','breathlessness','sweating','loss_of_appetite','mild_fever','yellowing_of_eyes','swelled_lymph_nodes','malaise','phlegm','blood_in_sputum'],
    'Common Cold':            ['continuous_sneezing','chills','fatigue','cough','high_fever','headache','swelled_lymph_nodes','malaise','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','loss_of_smell','muscle_pain'],
    'Pneumonia':              ['chills','fatigue','cough','high_fever','breathlessness','sweating','malaise','phlegm','rusty_sputum','fast_heart_rate'],
    'Dimorphic hemorrhoids':  ['constipation','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus'],
    'Heart attack':           ['vomiting','breathlessness','sweating','chest_pain','fast_heart_rate'],
    'Varicose veins':         ['fatigue','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','prominent_veins_on_calf'],
    'Hypothyroidism':         ['fatigue','weight_gain','cold_hands_and_feets','mood_swings','lethargy','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','depression','irritability','abnormal_menstruation'],
    'Hyperthyroidism':        ['fatigue','mood_swings','weight_loss','restlessness','sweating','diarrhoea','fast_heart_rate','excessive_hunger','muscle_weakness','irritability','abnormal_menstruation'],
    'Hypoglycemia':           ['fatigue','vomiting','anxiety','sweating','headache','nausea','blurred_and_distorted_vision','excessive_hunger','slurred_speech','irritability','drying_and_tingling_lips','palpitations'],
    'Osteoarthritis':         ['joint_pain','neck_pain','knee_pain','hip_joint_pain','swelling_joints','painful_walking'],
    'Arthritis':              ['muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','loss_of_balance'],
    '(Vertigo) Paroxysmal Positional Vertigo': ['vomiting','headache','nausea','spinning_movements','loss_of_balance','unsteadiness'],
    'Acne':                   ['skin_rash','pus_filled_pimples','blackheads','scurring'],
    'Urinary tract infection':['burning_micturition','bladder_discomfort','foul_smell_of_urine','continuous_feel_of_urine'],
    'Psoriasis':              ['skin_rash','joint_pain','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails'],
    'Impetigo':               ['skin_rash','high_fever','blister','red_sore_around_nose','yellow_crust_ooze'],
}

def generate_dataset(samples_per_disease=120):
    rows = []
    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        for _ in range(samples_per_disease):
            row = {s: 0 for s in SYMPTOMS}
            n_core = max(2, len(symptoms) - 2)
            chosen = np.random.choice(symptoms, size=n_core, replace=False)
            for s in chosen:
                if s in row:
                    row[s] = 1
            noise = np.random.choice(
                [s for s in SYMPTOMS if s not in symptoms],
                size=np.random.randint(0, 3), replace=False
            )
            for s in noise:
                row[s] = 1
            row['disease'] = disease
            rows.append(row)
    return pd.DataFrame(rows)

def train():
    print("Generating dataset...")
    df = generate_dataset()

    X = df[SYMPTOMS]
    y = df['disease']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy: {acc*100:.2f}%")

    os.makedirs('ml', exist_ok=True)
    joblib.dump(clf, 'ml/model.pkl')
    joblib.dump(le, 'ml/label_encoder.pkl')
    joblib.dump(SYMPTOMS, 'ml/symptoms.pkl')
    print("Model saved to ml/")

if __name__ == '__main__':
    train()