#%%
import pandas as pd
import numpy as np

#%%
# Added iucr index of crime code for Illinois
crimes = pd.read_csv('Crimes.csv')
iucr = pd.read_csv('IUCR.csv')
weather = pd.read_csv('weather.csv')
full_moon = pd.read_csv('full_moon.csv')
holidays = pd.read_csv('holidays.csv')

#%%
crimes.shape
#%%
# Use IUCR to mark which crimes were violent
crimes = pd.merge(crimes, iucr[['IUCR', 'isViolent']], on='IUCR', how='left')

#%%
# Add back False values for Nan
crimes['isViolent'].fillna(False, inplace=True)

#%%
# Drop timestamp on 
crimes['Date'] = pd.to_datetime(crimes['Date']).dt.date
crimes = crimes[crimes['Date'] >= pd.Timestamp('2010-01-01')]
crimes.reset_index(drop=True, inplace=True)

#%% 
crimes.head()
crimes.shape
#%%
# Count number of crimes by date
ncrimes = crimes.groupby('Date')['isViolent'].sum().reset_index()

#%%
ncrimes.head()

# %%
# Add weather variables to ncrimes
weather['datetime'] = pd.to_datetime(weather['datetime']).dt.date
weather.head()

#%% 
combine_data = pd.merge(ncrimes, weather, left_on='Date', right_on='datetime', how='left')
combine_data.drop(columns=['datetime'], inplace=True)
# Fill empty values with 0 to make sure everything is numerical
combine_data.fillna(0, inplace=True)

#%%
# Add Holidays
holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date
combine_data = pd.merge(combine_data, holidays, left_on='Date', right_on='Date', how='left')
combine_data['isHoliday'] = 0
combine_data.loc[combine_data['Holiday'].notnull(), 'isHoliday'] = 1
combine_data.head()

#%%
# Add Full Moon
full_moon['Date'] = pd.to_datetime(full_moon['Date']).dt.date
combine_data = pd.merge(combine_data, full_moon, left_on='Date', right_on='Date', how='left')
combine_data.head()

#%%
combine_data.fillna(0, inplace=True)
#combine_data.to_csv('clean_data.csv', index=False)


# %%
