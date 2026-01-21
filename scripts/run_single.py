import sys
import os
import logging
from pprint import pprint
from typing import cast
import pandas as pd

# Configure logging to display INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

sys.path.append(os.getcwd())

from src.main import create_workflow
from src.schemas.input_schema import InputSchema
from src.state.agent_state import AgentState

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# DATA_PATH = os.path.join(project_root, "data/gold_standard_dataset.csv")

# messages = pd.read_csv(DATA_PATH)

# Determine clinical case (Fake example)
fake_note = r"""
\nName:  ___               Unit No:   ___\n \n

Admission Date:  ___              Discharge Date:   ___
Date of Birth:  ___             Sex:   F
Service: MEDICINE

Allergies: 
No Known Allergies / Adverse Drug Reactions

Attending: ___

Chief Complaint:
Low oxygen saturation 

Major Surgical or Invasive Procedure:
PICC placement in left upper extremity

History of Present Illness:
HISTORY OF PRESENT ILLNESS: ___ yo F with history of 
bronchiectasis presents with hypoxia and productive cough. 

She was recently admitted in ___ (hospitalized from ___ 
to ___ for pneumonia vs. bronchiectasis exacerbation, afebrile 
and without leukocytosis, found to have positive sputum cultures 
for multidrug resistant, Zosyn-sensitive e.Coli and 
Achromobacter organisms. She was started on 4.5 g Zosyn Q8H via 
PICC line in her left upper arm and discharged ___ to home 
where she completed her 10-day antibiotic course via 
self-administration. Her pulmonologist, Dr. ___ she 
responded well to her treatment. The patient reports having 
severe, frequent, loose bowel movements during antibiotic course 
that loperamide helped only slightly. She was supposed to begin 
28-day course of Tobromycin after finishing her antibiotics, but 
do to insurance approval issues, she was not able. She followed 
up with Dr. ___ on ___ and was found to have stable 
exertional dyspnea. However, the patient states that she has had 
persistent O2 sats ranging from 85%-90% despite home oxygen for 
several days prior to presentation, and experienced an 
increasingly productive cough with purulent sputum.  Her oxygen 
saturation at home this morning was 88%, prompting her to call 
Dr. ___ told her to go to the emergency room. She 
also endorses sleeping trouble last night, a slight headache 
today that she attributes to not being able to eat anything 
while in the ER all day, nocturia, and has had some rhinorrhea 
and congestion ('normal' baseline symptoms). She denies fever, 
chills, chest pain, dyspnea, and hemoptysis.

In the ED, initial vitals were:  98.5  118/63  87  22  92% on RA
Exam notable for diffuse, course inspiratory crackles; fine 
crackles and rhonchi over the RLL.
Labs notable for mild leukocytosis.

Imaging notable for:
CXR (___)- Persistent right basilar patchy opacification, 
also found on previous CXR dated ___, which may represent 
asymmetric cylindrical bronchiectasis; however, an underlying 
atypical pneumonia cannot be excluded. No pleural effusion, 
pneumothorax, or cardiomegaly. Pulmonary vasculature, 
mediastinal and hilar contours appear normal. Moderate hiatal 
hernia with air-fluid level is appreciated. Diffuse interstitial 
thickening with bronchial wall thickening and bronchiectasis, 
especially prominent at the lung bases bilaterally; consistent 
with previous findings on CT from OSH dated ___.  

Patient was given Vanc and Zosyn after sputum sample was 
obtained.\nPatient was discussed over the phone with Dr. ___ 
recommended she have: sputum C&S taken to identify current 
infective organism(s); vancomycin and zosyn started, given 
previous sputum sample with gram positive organisms; CXR and 
CBC; supplemental O2, chest physical therapy. She was alert and 
oriented x 4, received a 20-gauge peripheral IV in the right 
antecubital vein on a 4.5gm zosyn/1gm vanc drip, and has been 
96% O2 sat on 2L NC, with no respiratory distress.\nDecision was made to admit for pneumonia vs. bronchiectasis \nexacerbation.\nVitals notable for 87% oxygen saturation on room air upon \npresentation at the ED.   \n\nReview of systems:  \n(+) Per HPI  \n\n \nPast Medical History:\nPMHx:\n# Bronchiectasis  \n# Asthma\n# COPD\n# Significant scoliosis  \n# lactose intolerance  \n# Right rotator cuff tear  \n# GERD\n# acoustic neuroma surgery with Dr. ___ at ___ ___  \n# cataract surgery Dr. ___ ___  \n# surgery for prolapsed uterus ___ \n# Osteopenia s/p ___ yrs of Fosamax\n# Sleep apnea\n# osteoarthritis\n\nPSHx:\nBrain surgery ___ for acoustic neuroma at ___\nCataract surgery\nTonsillectomy\n\n \nSocial History:\n___\nFamily History:\nFHx:\nRelative  Status    Age Problem              Comments           \nMother    ___  ___  CARDIAC ARREST                          \n           
Ulcerative COLITIS\nFather    ___  ___  ESOPHAGEAL CANCER                       \n           

             PROSTATE CANCER                         \nSister    Living    ___  CROHNS                    

              \nSon       Living    ___  IRRITABLE BOWEL SYNDROME\n
              Daughter  Living    ___  CHRONIC LYME DISEASE                    \nDaughter  Living    ___  \n \nPhysical Exam:\nADMISSION PHYSICAL EXAM:  \n==========================\nVS: 98.2  125/65  88  18  95% RA  ___ \nGen: Alert & oriented x 3. No apparent distress.\nHEENT: No oral or nasal ulcers. No cervical lymphadenopathy or \nthyromegaly. \nCV: Distant heart sounds. S1/S2. No murmurs, rubs, or gallops \nappreciated.\nPulm: Dry crackles and rhonchorous breath sounds bilaterally. \nAbd:  Benign. Nontender. Nondistended. No rebound or guarding. \nBowel sounds present and normoactive. \nGU: No suprapubic tenderness. No foley.\nExt: Heberden and ___ nodes in the DIP/PIP joints of the \nhands.  No lower extremity pitting edema. \nNeuro: Cranial nerves, strength, sensation grossly intact.\nSkin: No rashes.\nPsych: Pleasant mood. Oriented to person, place, and time.\n\nDISCHARGE AXAM \n===\nPhysical exam:\nVS: Tc 97.9 Tm 98.2  99-118/40-65  ___  18 95% 2L   Pain: ___\nGENERAL: NAD, alert, interactive, thin.\nHEENT: NC/AT, some watery tearing from left eye, sclerae \nanicteric, mucous membranes appear moist. No oral or nasal \nulcers. No cervical lymphadenopathy or thyromegaly. \nLUNGS: Diffuse, dry crackles and rhonchorous breath sounds \nbilaterally.\nHEART: Distant heart sounds. S1/S2. No murmurs, rubs, or gallops \nappreciated.\nABDOMEN: NABS, soft/NT/ND. Benign exam.\nGU: No suprapubic tenderness. No foley.\nEXTREMITIES: WWP. Heberden and ___ nodes in the DIP/PIP \njoints of the hands. PICC line in left upper extremity.  No \nlower extremity pitting edema. \nNEURO: awake, A&Ox3. Cranial nerves, strength, sensation grossly \nintact.\nPSYCH: Pleasant mood.\nSKIN: No rashes.\n \nPertinent Results:\n===\nADMISSION LABS \n===\n\n___ 01:26PM BLOOD WBC-11.8*# RBC-4.14 Hgb-12.5 Hct-40.8 \nMCV-99* MCH-30.2 MCHC-30.6* RDW-13.1 RDWSD-47.0* Plt ___\n___ 01:26PM BLOOD Neuts-71.9* Lymphs-17.1* Monos-8.3 \nEos-1.9 Baso-0.4 Im ___ AbsNeut-8.45* AbsLymp-2.01 \nAbsMono-0.98* AbsEos-0.22 AbsBaso-0.05\n___ 01:26PM BLOOD Plt ___\n___ 01:26PM BLOOD Glucose-103* UreaN-21* Creat-0.6 Na-139 \nK-4.3 Cl-101 HCO3-32 AnGap-10\n\n===\nMICRO\n=== \n___ Blood cultures - no growth 2 days \n\n___ sputum - Gram stain shows Gram positive cocci in pairs and \ngram and gram positive rods\n     culture - Gram negative rods - sparse growth. \n\n___ MRSA SCREEN - pending\n\n===\nIMAGING \n===\n___ CHEST PA LAT \nIMPRESSION: \n1. Persistent right basilar patchy opacification, which may \nrepresent \nasymmetric cylindrical bronchiectasis, better demonstrated on \nthe prior CT, \nhowever an underlying atypical pneumonia cannot be excluded. \n2. Hiatal hernia. \n\n___ CHEST PORTABLE PICC \nIMPRESSION:  \nNew left-sided PICC line.  The course is unremarkable, the tip \nprojects over the lower SVC.  No change in appearance of the \nlung parenchyma, including the  nodular opacities at the right \nlung basis. \n\n===\nDISCHARGE LABS\n===\n___ 09:50AM BLOOD WBC-10.2* RBC-3.82* Hgb-11.3 Hct-37.3 \nMCV-98 MCH-29.6 MCHC-30.3* RDW-13.2 RDWSD-47.5* Plt ___\n___ 06:20AM BLOOD Plt ___\n___ 06:20AM BLOOD Glucose-70 UreaN-13 Creat-0.7 Na-141 \nK-4.1 Cl-102 HCO3-27 AnGap-___ yo female with history of bronchiectasis, recurrent upper and \nlower respiratory infections, recent admission in ___ for \npneumonia vs. bronchiectasis found to have multidrug-resistant \nZosyn-sensitive e.Coli and Achromobacter treated successfully \nwith a 10-day course of Zosyn, presents with one week of \nproductive cough and hypoxia.\n\n#Hypoxemia: Concern for bronchiectasis exacerbation vs bacterial \npneumonia. She had slight leukocytosis of 11.8 on presentation \nbut afebrile. CXR showed right basilar patchy opacification, \nsimilar to previous CXR dated ___, possibly representing \nasymmetric cylindrical bronchiectasis; though an underlying \natypical pneumonia could not be excluded. Decision was made to \ncover broadly with vancomycin and zosyn given history of \nmultiple antibiotic resistant bugs in previous sputum samples. \nOn ___ Zosyn was reduced to 2.25g Q6H from 4.5g per pharmacy \nrecs. A PICC line was placed on ___. MRSA screen is still \npending and should result ___. Sputum culture gram stain showed \ngram + cocci in pairs and gram - rods. On ___ Sputum culture \nshowed sparse growth of only gram negative rods. Upon discharge \nshe was afebrile and WBC was 10.2. ___ blood cultures still \npending.\n\n#Bronchiectasis- She was prescribed how Ipratropium-Albuterol \nInhalation Spray and and Fluticasone Propionate inhaler. \nAcapella valve and home respiratory vest where also prescribed. \nHer oxygen saturation remained in the low to high ___ on 2L \nNasal cannula throughout hospital course. \n\n#Diarrhea- After start of antibiotics she reported multiple \nepisodes of diarrhea which happened to her during the last \ncourse of antibiotics as well. She was prescribed Loperamide 2 \nmg PO/NG Q3H:PRN diarrhea, with 4mg initial dose, which improved \nher symptoms to single bowel movements overnight. \n\nCHRONIC ISSUES: \n==========================\n#GERD:\n-continued home omeprazole\n\nCODE STATUS: DNR but ok to intubate.\n\nCONTACT: ___ (husband), ___ \n___\n\n===\nTransitional issues \n===\nPlease:\n-Continue 14 day course of Vancomycin 750 mg IV Q24 and \nPiperacillin-Tazobactam 2.25 IV Q6H (last day ___. Dr \n___ department) has agreed to manage her \nantibiotics after discharge, and will decide whether or not to \ncontinue Vancomycin, or to extend or shorten the duration of \nantibiotics\n-F/U MRSA screen and adjust use of Vancomycin as indicated \n-Follow up blood cultures and Sputum cultures \n-Assess for return of diarrhea and consider additional agents to \nhelp control symptoms. \n\n \nMedications on Admission:\nThe Preadmission Medication list is accurate and complete.\n1. Flovent HFA (fluticasone) 220 mcg/actuation inhalation BID \n2. Ipratropium-Albuterol Inhalation Spray 1 INH IH UP TO 5X PER \nDAY \n3. Acetaminophen 325 mg PO Q6H:PRN Pain - Mild \n4. Omeprazole 20 mg PO DAILY \n5. Aquoral (saliva substitute combo no.3) 2 sprays  mucous \nmembrane TID:PRN mucous  \n6. TraMADol 50 mg PO Q6H:PRN Pain - Moderate \n7. TraZODone 50 mg PO QHS:PRN insomnia \n8. Vitamin D 1000 UNIT PO 3X/WEEK (___) \n9. LOPERamide 2 mg PO Q6H:PRN diarrhea  \n10. Vitamin B Complex 1 CAP PO Frequency is Unknown \n11. Multivitamins 1 TAB PO DAILY \n\n \nDischarge Medications:\n1.  Piperacillin-Tazobactam 2.25 g IV Q6H \nplanned course for 14 days (final day ___, to be determined \nby Dr. ___ \nRX *piperacillin-tazobactam 2.25 gram 2.25 g IV every six (6) \nhours Disp #*46 Vial Refills:*0 \n2.  Vancomycin 750 mg IV Q 24H \nplanned 14 day course (final day ___, to be determined by \nDr. ___ \nRX *vancomycin 750 mg 750 mg IV Q24H Disp #*11 Vial Refills:*0 \n3.  Aquoral (saliva substitute combo no.3) 2 sprays  mucous \nmembrane TID:PRN mucous   \n4.  Flovent HFA (fluticasone) 220 mcg/actuation inhalation BID  \n5.  Ipratropium-Albuterol Inhalation Spray 1 INH IH UP TO 5X PER \nDAY  \n6.  LOPERamide 2 mg PO Q6H:PRN diarrhea  \nRX *loperamide 2 mg 1 capsule by mouth Q6H:PRN Disp #*30 Capsule \nRefills:*0 \n7.  Multivitamins 1 TAB PO DAILY  \n8.  Omeprazole 20 mg PO DAILY  \n9.  TraMADol 50 mg PO Q6H:PRN Pain - Moderate  \n10.  TraZODone 50 mg PO QHS:PRN insomnia  \n11.  Vitamin B Complex 1 CAP PO DAILY  \n12.  Vitamin D 1000 UNIT PO 3X/WEEK (___)  \n\n \nDischarge Disposition:\nHome With Service\n \nFacility:\n___\n \nDischarge Diagnosis:\nPRIMARY DIAGNOSES\n====================\nHealth ___ associated pneumonia \nBronchiectasis \n\n==\nSecondary \n===\nAntibiotic associated diarrhea \n\n \nDischarge Condition:\nMental Status: Clear and coherent.\nLevel of Consciousness: Alert and interactive.\nActivity Status: Ambulatory - Independent.\n\n \nDischarge Instructions:\nDear Ms. ___, \n    You were admitted to the hospital because you had low oxygen \nsaturation in your blood. It is likely due to a combination of \npneumonia and exacerbation of your bronchiectasis. We have \nstarted you on antibiotics called Vancomycin and Zosyn. While \nyou were in the hospital you received placement of a PICC line \nin your left arm which will be used to administer the \nantibiotics. The Zosyn will be continued for a total of 14 days \nending on ___. The vancomycin will be continued for now as \nwell. Dr. ___ will determine if it will be necessary to \ncontinue for 14 total days. Visiting nurses will be responsible \nfor administering the medication to you. \n\nYou were also given  Loperamide (also called Imodium) to lessen \nthe diarrhea symptoms that can be caused by the antibiotics. If \nyou continue to experience these symptoms please let Dr. \n___. \n\nPlease follow up with Dr. ___ ___ weeks for further \nmanagement of the antibiotics.\n\nRegards, \n\nYour ___ ___ Team\n \nFollowup Instructions:\n___\n
"""

app = create_workflow()

def main():
    print("🚀 Starting TriaLogic Single Case Test...")

    # 2. Create input object
    patient_input = InputSchema(
        subject_id=17032657,     # ID Fictício
        hadm_id=23389498,         # Ainda não internou
        raw_text=fake_note
    )

    # 3. Invoke the graph
    initial_state = cast(AgentState, {
        "input": patient_input,
        "extracted_data": None,
        "validation_errors": [],
        "validation_messages": [],
        "attempts": 0,
        "risk_score_report": None,
        "search_query": None,
        "context_category": None,
        "context_text": None,
        "auditor_report": None,
        "next_step": None,
        "plan": None
        })
    
    try:
        final_state = app.invoke(initial_state)

        print("\n" + "="*50)
        print("✅ FINISHED EXECUTION")
        print("="*50)

        print(f"\n🩺 Calculated Report:\n{final_state.get('risk_score_report')}")
        
        print(f"\n📚 Query RAG used:\n{final_state.get('search_query')}")
        
        print(f"\n⚖️ Auditor (Synthesizer) Report:")
        pprint(final_state.get('auditor_report'))

    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == '__main__':
    main()