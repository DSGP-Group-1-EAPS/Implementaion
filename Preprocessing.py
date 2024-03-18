import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineering(cleaned_df, data_for_MonthltDeptTotal):
  total_absence_by_main_dept = data_for_MonthltDeptTotal.groupby(['Encoded Department', 'LeaveYear','LeaveMonth']).size().reset_index(name='totalAbsence')
  jumper_team_monthly_leaves = {}
  mat_team_monthly_leaves = {}
  sewing_team_monthly_leaves = {}

  for index, row in total_absence_by_main_dept.iterrows():
    year = row['LeaveYear']
    month = row['LeaveMonth']
    if row['Encoded Department'] == 0:
        jumper_team_monthly_leaves[(year, month)] = row['totalAbsence']
    elif row['Encoded Department'] == 1:
        mat_team_monthly_leaves[(year, month)] = row['totalAbsence']
    elif row['Encoded Department'] == 2:
        sewing_team_monthly_leaves[(year, month)] = row['totalAbsence']

  print(sewing_team_monthly_leaves[2023,9])
  print(mat_team_monthly_leaves[2022,1])
  cleaned_df['MonthlyDeptTotal'] = 0
  last_leave_month = cleaned_df.iloc[-1]['LeaveMonth']
  for index, row in cleaned_df.iterrows():
    year = row['LeaveYear']
    month = row['LeaveMonth']
    if month == 12:
      next_year = row['LeaveYear']+1
      next_month = 1
    else:
      next_year = row['LeaveYear']
      next_month =  row['LeaveMonth']+1
    if next_year ==2024:
      print("broke on 2024")
      break
    if next_year ==2023 and next_month == last_leave_month+1:
      print("Broke on 2023-lastmonth")
      break
    if row['Encoded Department'] == 0:
      cleaned_df.at[index, 'MonthlyDeptTotal'] = jumper_team_monthly_leaves[(next_year, next_month)]
    elif row['Encoded Department'] == 1:
      cleaned_df.at[index, 'MonthlyDeptTotal'] = mat_team_monthly_leaves[(next_year, next_month)]
    elif row['Encoded Department'] == 2:
      cleaned_df.at[index, 'MonthlyDeptTotal'] = sewing_team_monthly_leaves[(next_year, next_month)]


  # Create a dictionary to store leave dates for each employee
  employee_leave_dates = {}

  # Iterate through the DataFrame to populate the dictionary
  for index, row in cleaned_df.iterrows():
      employee_code = row['Code']
      leave_date = row['Date']

      # Check if the employee code is already in the dictionary
      if employee_code in employee_leave_dates:
          # Append the leave date to the existing array
          employee_leave_dates[employee_code].append(leave_date)
      else:
          # Create a new array for the employee code
          employee_leave_dates[employee_code] = [leave_date]

  cleaned_df['LeavesNextMonth'] = 0

  for index, row in cleaned_df.iterrows():
      employee_code = row['Code']
      leave_year = row['LeaveYear']
      leave_month = row['LeaveMonth']

      if employee_code in employee_leave_dates:
          leave_years_months = [(date.year, date.month) for date in employee_leave_dates[employee_code]]

          if leave_month == 12:
              next_month = 1
              next_year = leave_year + 1
          else:
              next_month = leave_month + 1
              next_year = leave_year

          if (next_year, next_month) in leave_years_months:
              leaves_next_month = leave_years_months.count((next_year, next_month))
              cleaned_df.at[index, 'LeavesNextMonth'] = leaves_next_month

  cleaned_df['TargetCategory'] = pd.cut(cleaned_df['LeavesNextMonth'], bins=[-1, 3,float('inf')],
                                labels=['A', 'B'], right=False)

  return cleaned_df


def get_last_month(df):
    return df.iloc[-1]['LeaveMonth']

def remove_features(df):
    features_to_remove = ['TargetCategory', 'LeavesNextMonth', 'MonthlyDeptTotal']
    return df.drop(features_to_remove, axis=1)