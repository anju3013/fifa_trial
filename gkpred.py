import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency,chi2

# Function to categorize player positions
def categorize_position(positions):
    position = positions.split(', ')[0]  # Split positions if multiple
    if 'GK' in position:
        return 'GK'
    elif any(pos in position for pos in ['LCB','RCB', 'LB', 'RB', 'LWB', 'RWB', 'CB','DEF']):
        return 'DEF'
    elif any(pos in position for pos in ['CAM','LCM', 'RCM', 'CDM', 'LDM', 'RDM', 'CM', 'LM','RM', 'LAM', 'RAM','MID']):
      return 'MID'
    elif any(pos in position for pos in ['LS', 'ST','RS', 'LW', 'RW','CF', 'RF', 'LF', 'FW']):
        return 'FW'
    else:
        return 'Other'


# Function to categorize body types
def categorize_body_type(body_type):
  if 'Lean' in body_type:
    return 'Lean'
  elif 'Stocky' in body_type:
    return 'Stocky'
  elif 'Normal' in body_type:
    return 'Normal'
  elif 'Messi' in body_type:
    return 'Messi'
  elif 'C. Ronaldo' in body_type:
    return 'C. Ronaldo'
  elif 'Akinfenwa' in body_type:
    return 'Akinfenwa'
  elif 'Shaqiri' in body_type:
    return 'Shaqiri'
  elif 'Neymar' in body_type:
    return 'Neymar'
  elif 'Mohamed Salah' in body_type:
    return 'Mohamed Salah'
  elif 'Courtois' in body_type:
    return 'Courtois'
  else:
    return 'Other'

def chi2_test(df1, df2):
  alpha = 0.05
  cont_table = pd.crosstab(index=df1,columns=df2)
  #cont_table.head()
  # chi2 value, p value, degree of freedom , expected_table
  chi2_value, p, dof, expected_table = chi2_contingency(cont_table)

  print(f'chi2 value: {chi2_value}')
  print(f'p value: {p}')
  print(f'degree of freedom: {dof}')
  if p <= alpha:
    print(f'Reject null hypothesis. There exist some relation between features')
  else:
    print(f'Accept null hypothesis. Two features are not related')

#function to display null value percentage
def get_miss_percent(null_to_handle, df):
  percent_missing = []
  col_name = []
  for col in null_to_handle:
    percent_missing.append(df[col].isna().sum() * 100 / len(df[col]))
    col_name.append(col)
  missing_value_df = pd.DataFrame({'column_name': col_name,
                                 'percent_missing': percent_missing})
  return missing_value_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



df = pd.read_csv("players_21.csv")

df.head(30)

null_to_handle=[]
for col in df.columns:
  if df[col].isna().sum() > 0:
    null_to_handle.append(col)
print(f"columns to handle null values: {null_to_handle}")

get_miss_percent(null_to_handle, df)

#column dropped as so many null values 'player_tags','defending_marking'
df.drop(['player_tags','defending_marking'],axis=1,inplace=True)
null_to_handle.remove('player_tags')
null_to_handle.remove('defending_marking')

#columns dropped as they are id values 'club_name', 'league_name','nation_position', 'nation_jersey_number','team_jersey_number'
df.drop(['club_name','nation_position','team_jersey_number', 'nation_jersey_number'],axis=1,inplace=True)
null_to_handle.remove('club_name')
null_to_handle.remove('nation_position')
null_to_handle.remove('nation_jersey_number')
null_to_handle.remove('team_jersey_number')

#columns dropped as they derived from target column 'release_clause_eur'
df.drop(['release_clause_eur'],axis=1,inplace=True)
null_to_handle.remove('release_clause_eur')

#column to handle not affecting overall 'loaned_from', 'joined', 'contract_valid_until'
df.drop(['loaned_from','joined', 'contract_valid_until'],axis=1,inplace=True)
null_to_handle.remove('joined')
null_to_handle.remove('contract_valid_until')
null_to_handle.remove('loaned_from')

#We found that
#* 'pace','shooting','passing','dribbling','defending','physic' have null values for player_position = 'GK'
#* 'gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning' have null values for all other player_position

#**Conclusion: We need to handle and select features based on player_position**

#filling those columns with 0
for col in ['pace','shooting','passing','dribbling','defending','physic','gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning']:
  df[col].fillna(0,inplace=True)
  null_to_handle.remove(col)

null_to_handle

print("Rows with null laegue_rank: ", df.loc[df['league_rank'].isnull(),['league_name','team_position']].index.size)
df.loc[df['league_rank'].isnull(),['league_name','team_position']]

#**Conclusion**: League_rank and team_position are given null only where there is no league_name. We can drop those rows (225 rows) from prediction dataset (to confirm)

df.dropna(subset=['league_rank'], inplace = True)

df.isna().sum()[df.isna().sum() != 0]

#data split to 2 sets: 1. player_traits = null 2. not null

null_df = df[df['player_traits'].isnull()]
non_null_df = df[df['player_traits'].notnull()]

#To start with data preprocessing, deleting insignificant features from dataset

final_withplayertraits = non_null_df.drop(['sofifa_id', 'player_url', 'short_name', 'long_name', 'age', 'dob', 'height_cm',
       'weight_kg', 'nationality', 'league_name','potential',
       'value_eur', 'wage_eur','real_face','attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
       'attacking_short_passing', 'attacking_volleys','skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
       'movement_sprint_speed', 'movement_agility','movement_balance',
       'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
       'mentality_vision', 'mentality_penalties','defending_standing_tackle', 'defending_sliding_tackle',
       'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
       'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb',
       'rb'], axis = 1) #to remove after confirmation

final_withoutplayertraits = df.drop(['sofifa_id', 'player_url', 'short_name', 'long_name', 'age', 'dob', 'height_cm',
       'weight_kg', 'nationality', 'league_name','potential',
       'value_eur', 'wage_eur','real_face','attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
       'attacking_short_passing', 'attacking_volleys','skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
       'movement_sprint_speed', 'movement_agility','movement_balance',
       'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
       'mentality_vision', 'mentality_penalties','defending_standing_tackle', 'defending_sliding_tackle',
       'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
       'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb',
       'rb', 'player_traits'], axis = 1)

print(final_withplayertraits.shape)
print(final_withoutplayertraits.shape)

#**Feature reduction and optimization**

#Team_position is a subset of player_positions.
#* player_positions- potential versatility of players
#* team_position- actual role assigned in a team setup/match

#Hence choosing player_positions over team_position

#**Conclusion**: We can drop team_position and group player_positions to sub categories.

final_withoutplayertraits["player_positions"].value_counts().head(30)

# Apply categorization function to 'Position' column
final_withoutplayertraits['grouped_position'] = final_withoutplayertraits['player_positions'].apply(categorize_position)

print(final_withoutplayertraits[['player_positions','grouped_position']].head(10))

#to ask: LW, CAM

final_withoutplayertraits['grouped_position'].value_counts()

final_withoutplayertraits.drop(['team_position'], axis = 1, inplace = True)

#checking values of body_type
df['body_type'].value_counts()

final_withoutplayertraits['body_type'] = final_withoutplayertraits['body_type'].apply(categorize_body_type)

print(final_withoutplayertraits['body_type'].value_counts())

final_withoutplayertraits.head(10)

final_withoutplayertraits.columns

df_gk = final_withoutplayertraits.loc[final_withoutplayertraits['grouped_position'] == 'GK',['preferred_foot','league_rank', 'overall', 'international_reputation', 'weak_foot',
        'work_rate', 'body_type', 'movement_reactions', 'mentality_composure','gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
       'gk_positioning', 'goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
       'goalkeeping_reflexes']]
df_gk.head(10)

#cross checking correlation  for numerical data
corr_data =df_gk[['movement_reactions', 'mentality_composure','gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
       'gk_positioning', 'goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
       'goalkeeping_reflexes', 'overall']]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot = True)
plt.show()

sns.clustermap(corr_data.corr(), annot = True)

#cross checking ch2 test for discrete numerical data
for col in ['league_rank', 'international_reputation', 'weak_foot',
        'work_rate', 'body_type','preferred_foot']:
  print("\nCh2 on ", col)
  chi2_test(df_gk[col],df_gk['overall'])

#**Selected features for 'GK'**:

#'league_rank', 'international_reputation', 'weak_foot',
        #, 'body_type', 'movement_reactions', 'goalkeeping_diving'(more correlation than gk_diving),
        #'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
       #'goalkeeping_reflexes'

final_gk = df_gk[['league_rank', 'international_reputation', 'weak_foot', 'body_type', 'movement_reactions',
                 'goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']]
final_gk_target = df_gk['overall']

final_gk.head(10)

final_gk.describe()

final_gk.select_dtypes(['object']).value_counts()

final_gk_mean = final_gk.copy()

#obtaining a common feature for all gk attributes using mean
final_gk_mean['GK_attribute'] = final_gk_mean[['goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking',
                                     'goalkeeping_positioning', 'goalkeeping_reflexes']].mean(axis = 1)

final_gk_mean.drop(['goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking',
                                     'goalkeeping_positioning', 'goalkeeping_reflexes'], axis = 1, inplace = True)

print("Input features: ",final_gk_mean.columns)

#Data Encodng

encoded_final_gk = final_gk_mean.copy()

#label encode object variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col_toencode = encoded_final_gk.select_dtypes(['object'])
for col in col_toencode.columns:
  encoded_final_gk[col] = le.fit_transform(encoded_final_gk[col])

encoded_final_gk.head(10)

#Scaling

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_gk = sc.fit_transform(encoded_final_gk)
scaled_gk = pd.DataFrame(scaled_gk)

#minmax scaling
#from sklearn.preprocessing import MinMaxScaler
#minmax = MinMaxScaler(feature_range = (0,1))
#x_mm = minmax.fit_transform(encoded_final_gk)
#x_mm = pd.DataFrame(x_mm)

#Normalization

#from sklearn.preprocessing import normalize
#x_norm = normalize(scaled_gk)
#norm_gk = pd.DataFrame(x_norm)
#norm_gk.describe()

#Prediction models

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_gk, final_gk_target, test_size=0.2, random_state=42)

#linear regression model
lm = LinearRegression()
lm_model = lm.fit(X_train.values, y_train)
y_pred = lm_model.predict(X_test)

print("R-squared (R2) score:", r2_score(y_test, y_pred))
print("Mean-squared error (MSE):", mean_squared_error(y_test, y_pred))
#got r2 = 0.9984 when all gk attributes were separately included

#random forest classifier
rf_cl = RandomForestClassifier(random_state = 1, n_estimators=20, max_depth = 20, criterion='entropy')
rf_cl.fit(X_train, y_train)
y_pred = rf_cl.predict(X_test)

print("R-squared (R2) score:", r2_score(y_test, y_pred))

#decision tree model
dt_cl = DecisionTreeClassifier(random_state =2)
dt_cl.fit(X_train, y_train)
y_pred = dt_cl.predict(X_test)

print("R-squared (R2) score:", r2_score(y_test, y_pred))

#outlier detection in all continuous variables (none here)
#check feature reduction using mean in GK attributes (corr(diving and reflexes) > 0.9)



import pickle
pickle.dump(rf_cl, open('model.pkl', 'wb'))

pickled_model=pickle.load(open('model.pkl','rb'))

pickled_model.predict([[1.0,3,3,3,88,87.4]])