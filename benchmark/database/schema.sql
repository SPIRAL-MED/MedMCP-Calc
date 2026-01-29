-- create sample schema and tables (ASCII-only comments)

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

DROP TABLE IF EXISTS public.patient_information;
CREATE TABLE public.patient_information (
  medical_institution_code TEXT, -- medical institution code
  patient_id                TEXT NOT NULL PRIMARY KEY, -- patient identifier
  campus_name               TEXT NOT NULL, -- campus name
  patient_name              TEXT NOT NULL, -- patient name
  gender                    TEXT NOT NULL, -- gender
  birth_date                DATE, -- birth date
  visit_card_no             TEXT, -- internal visit card no
  marriage_status           TEXT, -- marriage status
  ethnicity                 TEXT, -- ethnicity
  nationality               TEXT, -- nationality
  abo_blood_type            TEXT, -- ABO blood type
  rh_blood_type             TEXT, -- Rh blood type
  source_system             TEXT, -- source system
  is_valid                  BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.visit_inpatient;
CREATE TABLE public.visit_inpatient (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL PRIMARY KEY, -- visit identifier
  visit_type               TEXT NOT NULL, -- visit type (reference code)
  campus_name              TEXT NOT NULL, -- campus name
  inpatient_no             TEXT, -- source inpatient number
  visits_num               INTEGER, -- visit count
  patient_name             TEXT NOT NULL, -- patient name
  gender                   TEXT NOT NULL, -- gender
  patient_age              INTEGER, -- patient age
  patient_age_unit         TEXT, -- age unit
  admission_time           TIMESTAMP, -- admission time
  admission_way            TEXT, -- admission way e.g. outpatient, emergency
  admission_dept_name      TEXT, -- admission department name
  admission_dept_code      TEXT, -- admission department source code
  admission_ward_name      TEXT, -- admission ward name
  admission_ward_code      TEXT, -- admission ward source code
  major_diagnosis          TEXT, -- main diagnosis
  bed_no                   TEXT, -- bed number
  current_dept_name        TEXT, -- current department name
  current_dept_code        TEXT, -- current department source code
  current_ward_name        TEXT, -- current ward name
  current_ward_code        TEXT, -- current ward source code
  discharge_dept_name      TEXT, -- discharge department name
  discharge_dept_code      TEXT, -- discharge department source code
  discharge_ward_name      TEXT, -- discharge ward name
  discharge_ward_code      TEXT, -- discharge ward source code
  discharge_time           TIMESTAMP, -- discharge time
  create_time              TIMESTAMP, -- record create time
  update_time              TIMESTAMP, -- record update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.examination_report;
CREATE TABLE public.examination_report (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT NOT NULL PRIMARY KEY, -- report identifier
  campus_name              TEXT NOT NULL, -- campus name
  visit_type               TEXT NOT NULL, -- visit type
  patient_name             TEXT NOT NULL, -- patient name
  gender                   TEXT NOT NULL, -- gender
  patient_age              INTEGER, -- patient age
  patient_age_unit         TEXT, -- age unit
  bed_no                   TEXT, -- bed number
  clinic_diag_name         TEXT, -- clinical diagnosis
  medical_history          TEXT, -- medical history summary
  exam_type                TEXT, -- examination type
  exam_item_name           TEXT, -- exam item name
  exam_site                TEXT, -- exam site
  exam_method              TEXT, -- exam method
  exam_tech                TEXT, -- exam technique
  report_name              TEXT, -- report name
  enhance_scan_flag        TEXT, -- enhancement flag
  is_anesthesia            TEXT, -- anesthesia flag
  exam_time                TIMESTAMP, -- exam time
  exam_abnormal_flag       TEXT, -- abnormal flag (1 normal, 2 abnormal, 3 uncertain)
  exam_findings            TEXT, -- findings
  exam_conclusion          TEXT, -- conclusion
  radiology_note           TEXT, -- radiology notes
  intact_exam_report       TEXT, -- full report text
  summary_note             TEXT, -- summary description
  comment                  TEXT, -- comment
  review_note              TEXT, -- review notes
  application_no           TEXT, -- application number
  review_time              TIMESTAMP, -- review time
  report_time              TIMESTAMP, -- report time
  report_state             TEXT, -- report state (0 void, 1 normal)
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.laboratory_result;
CREATE TABLE public.laboratory_result (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT NOT NULL, -- report identifier
  report_item_id           TEXT NOT NULL PRIMARY KEY, -- report item identifier
  campus_name              TEXT NOT NULL, -- campus name
  visit_type               TEXT NOT NULL, -- visit type
  test_report_name         TEXT, -- test package name
  sort_no                  TEXT, -- display order
  test_method              TEXT, -- test method
  test_item_name           TEXT, -- test item name
  test_result              TEXT, -- test result
  unit                     TEXT, -- result unit
  sample_name              TEXT, -- sample type
  normal_low               FLOAT, -- normal low
  normal_high              FLOAT, -- normal high
  reference_range          TEXT, -- reference range
  abnormal_flag            TEXT, -- abnormal flag
  critical_low             FLOAT, -- critical low
  critical_high            FLOAT, -- critical high
  critical_flag            TEXT, -- critical flag
  absurd_low               FLOAT, -- absurd low
  absurd_high              FLOAT, -- absurd high
  report_time              TIMESTAMP, -- report time
  comment                  TEXT, -- comment
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.diagnostic_record;
CREATE TABLE public.diagnostic_record (
  medical_institution_code TEXT, -- medical institution code
  campus_name              TEXT NOT NULL, -- campus name
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT, -- document id
  report_item_id           TEXT NOT NULL PRIMARY KEY, -- diagnosis id
  diagnosis_class_name     TEXT, -- diagnosis class name
  diagnosis_code           TEXT, -- diagnosis source code
  diagnosis_seq            TEXT, -- diagnosis sequence
  diagnosis_name           TEXT, -- diagnosis name
  main_flag                TEXT, -- main diagnosis flag
  diagnosis_source         TEXT, -- source of diagnosis
  diagnosis_time           TIMESTAMP, -- diagnosis time
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.anethesia_record;
CREATE TABLE public.anethesia_record (
  medical_institution_code TEXT, -- medical institution code
  campus_name              TEXT NOT NULL, -- campus name
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT NOT NULL PRIMARY KEY, -- anesthesia record id
  visit_type               TEXT NOT NULL, -- visit type
  height                   FLOAT, -- height in cm
  weight                   FLOAT, -- weight in kg
  urine_volume             FLOAT, -- urine volume in ml
  blood_lossed             FLOAT, -- blood loss in ml
  asa_class_code           TEXT, -- ASA class
  surgical_position        TEXT, -- surgical position
  whether_fast             TEXT, -- fasting flag
  preoperative_diagnosis   TEXT, -- preoperative diagnosis
  propose_surgery_name     TEXT, -- proposed surgery name
  intraoperative_surgical_name TEXT, -- intraoperative surgery name
  breath_type              TEXT, -- breath type
  tracheal_intubation_type TEXT, -- intubation type
  drug_before_anesthesia   TEXT, -- pre-anesthesia drugs
  anesthesia_start_time    TIMESTAMP, -- start time
  anesthesia_end_time      TIMESTAMP, -- end time
  anesthesia_type          TEXT, -- anesthesia type
  anesthesia_real_duration FLOAT, -- real duration in hours
  go_after_operation       TEXT, -- post-op disposition
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.admission_record;
CREATE TABLE public.admission_record (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT NOT NULL PRIMARY KEY, -- document id
  inpatient_no             TEXT NOT NULL, -- inpatient number
  campus_name              TEXT NOT NULL, -- campus name
  patient_name             TEXT NOT NULL, -- patient name
  gender                   TEXT NOT NULL, -- gender
  admission_time           TIMESTAMP, -- admission time
  admission_dept_name      TEXT, -- admission department
  admission_dept_code      TEXT, -- admission department code
  admission_ward_name      TEXT, -- admission ward
  admission_ward_code      TEXT, -- admission ward code
  record_title             TEXT, -- record title
  document_content         TEXT, -- document content
  chief_complaints         TEXT, -- chief complaints
  present_illness          TEXT, -- present illness
  past_medical_history     TEXT, -- past medical history
  personal_history         TEXT, -- personal history
  marital_obstetrical_history TEXT, -- marital/obstetrical history
  family_history           TEXT, -- family history
  physical_examination     TEXT, -- physical exam
  accessory_examination    TEXT, -- auxiliary exams
  special_examination      TEXT, -- specialty exams
  intact_physical_examination TEXT, -- combined physical exam text
  admission_diagnosis      TEXT, -- admission diagnosis
  initial_diagnosis        TEXT, -- initial diagnosis
  modified_diagnosis       TEXT, -- modified diagnosis
  supplementary_diagnosis  TEXT, -- supplementary diagnosis
  discharge_diagnosis      TEXT, -- discharge diagnosis
  record_time              TIMESTAMP, -- record time
  record_state_name        TEXT, -- record state
  creator_name             TEXT, -- creator name
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.order_inpatient;
CREATE TABLE public.order_inpatient (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  report_id                TEXT NOT NULL PRIMARY KEY, -- order id
  inpatient_no             TEXT, -- inpatient number
  campus_name              TEXT NOT NULL, -- campus name
  patient_name             TEXT NOT NULL, -- patient name
  group_no                 TEXT, -- group number
  group_seq                TEXT, -- group sequence
  order_class              TEXT, -- order class
  order_type               TEXT, -- order type
  drug_flag                TEXT, -- is drug
  order_name               TEXT, -- order name
  specification            TEXT, -- specification
  dose_form                TEXT, -- dose form
  unit_price               FLOAT, -- unit price
  quantity                 FLOAT, -- quantity
  quantity_unit            TEXT, -- quantity unit
  once_dose                FLOAT, -- dose per time
  dose_unit                TEXT, -- dose unit
  frequency                TEXT, -- frequency
  special_execution_time   TEXT, -- special execution time
  special_execution_dose   TEXT, -- special execution dose
  usage                    TEXT, -- usage
  procedure_name           TEXT, -- procedure name
  herbal_payments          INTEGER, -- herbal payments
  execution_dept           TEXT, -- execution department
  skin_test                TEXT, -- skin test flag
  urgent_flag              TEXT, -- urgent flag (0 normal, 1 urgent)
  is_discharge_medicine    TEXT, -- discharge medicine flag
  order_advice             TEXT, -- order advice
  order_begin_time         TIMESTAMP, -- order begin time
  order_end_time           TIMESTAMP, -- order end time
  dept_name                TEXT, -- department name
  dept_code                TEXT, -- department code
  ward_name                TEXT, -- ward name
  ward_code                TEXT, -- ward code
  submit_dept_name         TEXT, -- submit department name
  submit_dept_code         TEXT, -- submit department code
  submit_time              TIMESTAMP, -- submit time
  body_site_name           TEXT, -- body site name
  sample_name              TEXT, -- sample name
  confirm_time             TIMESTAMP, -- confirm time
  cancel_time              TIMESTAMP, -- cancel time
  execution_time           TIMESTAMP, -- execution time
  order_state_name         TEXT, -- order state name
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);

DROP TABLE IF EXISTS public.vital_signs;
CREATE TABLE public.vital_signs (
  medical_institution_code TEXT, -- medical institution code
  patient_id               TEXT NOT NULL, -- patient identifier
  visit_id                 TEXT NOT NULL, -- visit identifier
  vs_id                    TEXT NOT NULL PRIMARY KEY, -- vital sign id
  campus_name              TEXT NOT NULL, -- campus name
  patient_name             TEXT NOT NULL, -- patient name
  dept_name                TEXT, -- department name
  dept_code                TEXT, -- department code
  ward_name                TEXT, -- ward name
  ward_code                TEXT, -- ward code
  bed_no                   TEXT, -- bed number
  plan_time                TIMESTAMP, -- planned time
  measure_time             TIMESTAMP, -- measured time
  breathe                  TEXT, -- respiration (per min)
  pulse                    TEXT, -- pulse (per min)
  heart_rate               TEXT, -- heart rate (per min)
  temperature              TEXT, -- temperature (C)
  systolic_pressure        TEXT, -- systolic pressure (mmHg)
  diastolic_pressure       TEXT, -- diastolic pressure (mmHg)
  height                   TEXT, -- height (cm)
  weight                   TEXT, -- weight (kg)
  defecate_frequency       TEXT, -- defecation frequency (per day)
  urine_volume             TEXT, -- urine volume (ml)
  sputum_volume            TEXT, -- sputum volume (ml)
  drainage_volume          TEXT, -- drainage volume (ml)
  emesis_volume            TEXT, -- emesis volume (ml)
  output_total_volume      TEXT, -- total output volume (ml)
  income_volume            TEXT, -- intake volume (ml)
  postoperative_days       INTEGER, -- postoperative days
  after_delivery_days      INTEGER, -- days after delivery
  days_in_hospital         INTEGER, -- days in hospital
  create_time              TIMESTAMP, -- create time
  update_time              TIMESTAMP, -- update time
  is_valid                 BOOLEAN -- is valid
);
