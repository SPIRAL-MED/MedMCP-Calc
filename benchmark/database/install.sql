-- psql -h localhost -p 5432 -U <your_user> -d <your_db> -v data_dir=<PATH TO DATA DIR> -f /full/path/to/load_sample.sql

-- Set the data directory (replace with your actual path)
\cd :data_dir

-- Set encoding to UTF-8
SET CLIENT_ENCODING TO 'utf8';

-- Load data
\COPY patient_information FROM PROGRAM 'gzip -dc patient_information.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY visit_inpatient FROM PROGRAM 'gzip -dc visit_inpatient.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY examination_report FROM PROGRAM 'gzip -dc examination_report.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY laboratory_result FROM PROGRAM 'gzip -dc laboratory_result.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY diagnostic_record FROM PROGRAM 'gzip -dc diagnostic_record.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY anethesia_record FROM PROGRAM 'gzip -dc anethesia_record.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY admission_record FROM PROGRAM 'gzip -dc admission_record.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY order_inpatient FROM PROGRAM 'gzip -dc order_inpatient.csv.gz' DELIMITER ',' CSV HEADER NULL '';
\COPY vital_signs FROM PROGRAM 'gzip -dc vital_signs.csv.gz' DELIMITER ',' CSV HEADER NULL '';

-- Display completion message
\echo 'All data has been loaded successfully!'