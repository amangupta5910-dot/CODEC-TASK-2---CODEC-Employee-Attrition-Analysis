import joblib
import pandas as pd

# Load Model
model = joblib.load("employee_attrition_model.pkl")

print("=" * 50)
print("      Employee Attrition Prediction")
print("=" * 50)

# Take Inputs
Age = int(input("Age: "))
BusinessTravel = int(input("BusinessTravel (0-2): "))
DailyRate = int(input("DailyRate: "))
Department = int(input("Department (0-2): "))
DistanceFromHome = int(input("DistanceFromHome: "))
Education = int(input("Education (1-5): "))
EducationField = int(input("EducationField (0-5): "))
EnvironmentSatisfaction = int(input("EnvironmentSatisfaction (1-4): "))
Gender = int(input("Gender (Female=0, Male=1): "))
HourlyRate = int(input("HourlyRate: "))
JobInvolvement = int(input("JobInvolvement (1-4): "))
JobLevel = int(input("JobLevel (1-5): "))
JobRole = int(input("JobRole (0-8): "))
JobSatisfaction = int(input("JobSatisfaction (1-4): "))
MaritalStatus = int(input("MaritalStatus (0-2): "))
MonthlyIncome = int(input("MonthlyIncome: "))
MonthlyRate = int(input("MonthlyRate: "))
NumCompaniesWorked = int(input("NumCompaniesWorked: "))
OverTime = int(input("OverTime (No=0, Yes=1): "))
PercentSalaryHike = int(input("PercentSalaryHike: "))
PerformanceRating = int(input("PerformanceRating (3-4): "))
RelationshipSatisfaction = int(input("RelationshipSatisfaction (1-4): "))
StockOptionLevel = int(input("StockOptionLevel (0-3): "))
TotalWorkingYears = int(input("TotalWorkingYears: "))
TrainingTimesLastYear = int(input("TrainingTimesLastYear: "))
WorkLifeBalance = int(input("WorkLifeBalance (1-4): "))
YearsAtCompany = int(input("YearsAtCompany: "))
YearsInCurrentRole = int(input("YearsInCurrentRole: "))
YearsSinceLastPromotion = int(input("YearsSinceLastPromotion: "))
YearsWithCurrManager = int(input("YearsWithCurrManager: "))

# Create DataFrame
data = pd.DataFrame([[

Age,
BusinessTravel,
DailyRate,
Department,
DistanceFromHome,
Education,
EducationField,
EnvironmentSatisfaction,
Gender,
HourlyRate,
JobInvolvement,
JobLevel,
JobRole,
JobSatisfaction,
MaritalStatus,
MonthlyIncome,
MonthlyRate,
NumCompaniesWorked,
OverTime,
PercentSalaryHike,
PerformanceRating,
RelationshipSatisfaction,
StockOptionLevel,
TotalWorkingYears,
TrainingTimesLastYear,
WorkLifeBalance,
YearsAtCompany,
YearsInCurrentRole,
YearsSinceLastPromotion,
YearsWithCurrManager

]], columns=[

'Age',
'BusinessTravel',
'DailyRate',
'Department',
'DistanceFromHome',
'Education',
'EducationField',
'EnvironmentSatisfaction',
'Gender',
'HourlyRate',
'JobInvolvement',
'JobLevel',
'JobRole',
'JobSatisfaction',
'MaritalStatus',
'MonthlyIncome',
'MonthlyRate',
'NumCompaniesWorked',
'OverTime',
'PercentSalaryHike',
'PerformanceRating',
'RelationshipSatisfaction',
'StockOptionLevel',
'TotalWorkingYears',
'TrainingTimesLastYear',
'WorkLifeBalance',
'YearsAtCompany',
'YearsInCurrentRole',
'YearsSinceLastPromotion',
'YearsWithCurrManager'

])

# Prediction
prediction = model.predict(data)

print("\n==============================")

if prediction[0] == 1:
    print("Employee is likely to LEAVE the company.")
else:
    print("Employee is NOT likely to leave the company.")

print("==============================")