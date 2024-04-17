#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('clean_data.csv')

# %%
data['Date'] = pd.to_datetime(data['Date'])

#%%
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Group by month and year, summing 'isViolent' for each day
m_violent_crimes = data.groupby(['Month', 'Year'])['isViolent'].sum().reset_index()
m_violent_crimes = m_violent_crimes.groupby(['Month'])['isViolent'].mean().reset_index()
m_violent_crimes

#%%
# Group 
h_violent_crimes = data.groupby(['Holiday', 'Year'])['isViolent'].sum().reset_index()
h_violent_crimes = h_violent_crimes.groupby(['Holiday'])['isViolent'].mean().reset_index()
h_violent_crimes

#%%
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
# Scatter plot for 'snow'
ax1.scatter(monthly_avg['Month'], monthly_avg['snow'], color='purple', label='Snow')
ax1.set_ylabel('Snow')
ax1.set_title('Monthly Averages: Winter Variables')

# Scatter plot for 'snowdepth'
ax2.scatter(monthly_avg['Month'], monthly_avg['snowdepth'], color='gray', label='Snowdepth')
ax2.set_xlabel('Month')
ax2.set_ylabel('Snow Depth')

for ax in axs3.flatten():
    # Set tick labels to display all values on the x-axis
    ax.set_xticks(monthly_avg['Month'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

#%%
#m_violent_crimes.to_csv('m_violent_crimes.csv', index=False)
#h_violent_crimes.to_csv('h_violent_crimes.csv', index=False)

#%%
holiday_crimes = pd.read_csv('h_violent_crimes.csv')
holiday_crimes
#%%
# Group by month, averaging all variables including 'isViolent'
monthly_avg = data.groupby(['Month']).mean().reset_index()
monthly_avg['isViolent'] = m_violent_crimes['isViolent']
monthly_avg

# %%
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bar plot for 'isViolent'
sns.barplot(x=monthly_avg['Month'], y=monthly_avg['isViolent'], data=monthly_avg, ax=ax, errorbar=None)

#%%

#%%
# Group by month, averaging all variables including 'Holiday'
monthly_avg = data.groupby(['Month']).mean().reset_index()
monthly_avg

# %%
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))





#%%
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the scatter plot for 'temp max'
sns.scatterplot(x=monthly_avg['Month'], y=monthly_avg['cloudcover'], data=monthly_avg, ax=ax, color='orange')
#sns.scatterplot(x=monthly_avg['Month'], y=monthly_avg['windspeed'], data=monthly_avg, ax=ax, color='red')
#sns.scatterplot(x=monthly_avg['Month'], y=monthly_avg['sealevelpressure'], data=monthly_avg, ax=ax, color='purple')

# Set labels and title
ax.set_xlabel('Number of Violent Crimes')
ax.set_ylabel('Month')
ax.set_title('Combination Plot of Violent Crimes and Feelslike Temp')

# Show plot
plt.show()

# %%
fig, axs1 = plt.subplots(6, 1, figsize=(10, 15))

# Scatter plot for 'Temp'
axs1[0].scatter(monthly_avg['Month'], monthly_avg['temp'], color='green', label='Temperature')
axs1[0].set_ylabel('Temperature')
axs1[0].set_title('Monthly Averages: Summer Variables')

# Scatter plot for 'Dew'
axs1[1].scatter(monthly_avg['Month'], monthly_avg['dew'], color='blue', label='Dew')
axs1[1].set_ylabel('Dew')

# Scatter plot for 'Solar Radiation'
axs1[2].scatter(monthly_avg['Month'], monthly_avg['solarradiation'], color='purple', label='Solar Radiation')
axs1[2].set_ylabel('Solar Radiation')

# Scatter plot for 'Solar Energy'
axs1[3].scatter(monthly_avg['Month'], monthly_avg['solarenergy'], color='magenta', label='Solar Energy')
axs1[3].set_ylabel('Solar Energy')

# Scatter plot for 'UVIndex'
axs1[4].scatter(monthly_avg['Month'], monthly_avg['uvindex'], color='red', label='UV Index')
axs1[4].set_ylabel('UV Index')

# Scatter plot for 'Visibility'
axs1[5].scatter(monthly_avg['Month'], monthly_avg['visibility'], color='orange', label='Visibility')
axs1[5].set_xlabel('Month')
axs1[5].set_ylabel('Visibility')

for ax in axs1.flatten():
    # Set tick labels to display all values on the x-axis
    ax.set_xticks(monthly_avg['Month'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

#%%
fig, axs2 = plt.subplots(4, 1, figsize=(10, 10))

# Scatter plot for 'feelslike'
axs2[0].scatter(monthly_avg['Month'], monthly_avg['precip'], color='green', label='Feels Like')
axs2[0].set_ylabel('Precipitation')
axs2[0].set_title('Monthly Averages: Misc Variables')

# Scatter plot for 'dew'
axs2[1].scatter(monthly_avg['Month'], monthly_avg['humidity'], color='blue', label='Humidity')
axs2[1].set_ylabel('Humidity')

# Scatter plot for 'dew'
axs2[2].scatter(monthly_avg['Month'], monthly_avg['cloudcover'], color='purple', label='Cloudcover')
axs2[2].set_ylabel('Cloud Cover')

axs2[3].scatter(monthly_avg['Month'], monthly_avg['windgust'], color='magenta', label='Wind Gust')
axs2[3].set_ylabel('Wind Gust')
axs2[3].set_xlabel('Month')

for ax in axs2.flatten():
    # Set tick labels to display all values on the x-axis
    ax.set_xticks(monthly_avg['Month'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()
#%%
X_axis = np.arange(len(holiday_crimes)) 
  
plt.bar(X_axis - 0.2, holiday_crimes['isViolent'], 0.4, label = 'Avg Crimes per Holiday') 
plt.bar(X_axis + 0.2, holiday_crimes['Daily Average per Month'], 0.4, label = 'Avg Crimes per Day in Same Month') 
  
plt.xticks(holiday_crimes['Holiday'], holiday_crimes) 
plt.xlabel("Groups") 
plt.ylabel("Average number of Violent Crimes") 
plt.title("Average Holiday v. Daily Crimes in a Month") 
plt.legend() 
plt.show() 



#for ax in axs3.flatten():   # Set tick labels to display all values on the x-axis#    ax.set_xticks(monthly_avg['Month'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

#%%
#%%
fig, axs4 = plt.subplots(2, 1, figsize=(10, 5))
# Scatter plot for 'snow'
axs4[0].scatter(monthly_avg['Month'], monthly_avg['isViolent'], color='purple', label='Snow')
axs4[0].set_ylabel('Snow')
axs4[0].set_title('Monthly Averages: Winter Variables')

# Scatter plot for 'snowdepth'
axs4[1].scatter(monthly_avg['Holiday'], monthly_avg['isViolent'], color='gray', label='Snowdepth')
axs4[1].set_xlabel('Month')
axs4[1].set_ylabel('Snow Depth')

#for ax in axs4.flatten():
    # Set tick labels to display all values on the x-axis
#    ax.set_xticks(monthly_avg['Month'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

# %%
# Bar plot for 'isViolent'
ax2.bar(monthly_avg['Month'], monthly_avg['isViolent'], color=None, alpha=0.5, label='Violent Crimes')
ax2.set_xlabel('Month')
ax2.set_ylabel('Violent Crimes')
ax2.set_title('Monthly Averages: Violent Crimes')

#%%
x = monthly_avg['Month']
y_variables = ['feelslike', 'dew',]

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['solarenergy',  'uvindex']

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['snow', 'snowdepth', ]

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, -monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['humidity']

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['windgust']

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['precip',]

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
x = monthly_avg['Month']
y_variables = ['visibility']

# Create a scatter plot for each y variable
plt.figure(figsize=(10, 6))
for y_var in y_variables:
    plt.scatter(x, monthly_avg[y_var], label=y_var)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Scatter plot of Monthly Averages')

# Add legend
plt.legend()

#%%
# Summer Variables
'feelslike', 'dew', 'solarenergy', 'solarradiation', 'uvindex', 'visibility', 
# Misc Variable
'humidity', 'windgust', 'precip'
# Winter Variables
'snow', 'snowdepth', 

#%%
'feelslike',	
'dew',	
'humidity',	
'precip',	
#'precipprob', 
#'precipcover',	
'preciptype',	# categorical
'snow',	
'snowdepth',	
'windgust',	
#'windspeed',	
#'winddir',	
#'sealevelpressure',	
#'cloudcover',	
'visibility',	
'solarradiation',	
'solarenergy',	
'uvindex',
'moonphase',
#'conditions', #categorical
'Holiday'

# %%
def waterfall_plot(categories, values):

    cum_values = np.cumsum(values)

    data = pd.DataFrame({'Categories': categories, 'Values': values, 'Cumulative Values': cum_values})

    plt.figure(figsize=(10, 6))
    plt.bar(data['Categories'].astype(str), data['Values'], width=0.5, align='center', alpha=0.7, color='blue', label='Features')

    plt.xlabel('Holidays')
    plt.ylabel('Feature Importance')
    plt.title('Distribution of Feature Importance by Holiday')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

waterfall_plot(holiday_features['Feature Importance for Holidays*'], holiday_features['Unnamed: 1'])
