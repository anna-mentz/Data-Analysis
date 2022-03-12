import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.interpolate import interp1d


data=pd.read_excel("weather_data.xlsx", header=0, na_values=' ')
df=pd.DataFrame(data)
df.info()
df.isnull().values.any()
df.info()
df['HIGH'] = df['HIGH'].map(lambda x: float(x))
df['LOW'] = df['LOW'].map(lambda x: float(x))
df['TEMP'] = df['TEMP'].map(lambda x: float(x))
df['HDD'] = df['HDD'].map(lambda x: float(x))
df['CDD'] = df['CDD'].map(lambda x: float(x))
df['RAIN'] = df['RAIN'].map(lambda x: float(x))
df['WINDHIGH'] = df['WINDHIGH'].map(lambda x: float(x))

"""Question1:"""
#Fill NaN values with 'DEC' in column MONTH
df['MONTH'].isnull().values.any()
df['MONTH'].isnull().sum()
df["MONTH"].fillna("DEC", inplace=True)
df['MONTH'].isnull().sum()
print(df['MONTH'])

#Finding NaN values with CUBIC SPLINES 
df['HIGH'].isnull().values.any()
df['HIGH'].isnull().sum()
df['LOW'].isnull().values.any()
df['LOW'].isnull().sum()

#to find the 3 before and 3 after values
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]

print(rows_with_NaN) #rows are 17, 55, 61, 103


#for row 17 -HIGH
print(df.iloc[14:21,1:4])
x=[15, 16, 17, 19, 20, 21]
y=[12.7, 12.3, 12.5, 12.9, 7.8, 9.2]

f1=interp1d(x,y, kind='cubic')
print(f1(18))

#for row 55 - LOW
print(df.iloc[52:59,1:6])
x=[22, 23, 24, 26, 27, 28]
y=[7.9, 9.2, 11.0, 11.6, 9.6, 9.1]

f1=interp1d(x,y, kind='cubic')
print(f1(25))

#for row 61 - HIGH
print(df.iloc[58:65,1:6])
x=[1, 2, 4, 5, 6, 28]
y=[18.3, 15.4, 18.2, 19.0, 17.9, 18.1]

f1=interp1d(x,y, kind='cubic')
print(f1(3))

#for row 103 - LOW
print(df.iloc[100:107,1:6])
x=[11, 12, 13, 15, 16, 17]
y=[11.2, 11.4, 12.2, 13.4, 12.6, 11.8]

f1=interp1d(x,y, kind='cubic')
print(f1(14))

#END QUESTION 1, prepartion for QUESTION 2

#Filling the NaN values with the values found from interpolation
#(values: 18 --> 13.91, 25 -->12.07, 3-->16.24, 14 --> 13.11)

#row:17 value:13.91 column:HIGH 3 
df.iloc[17,3]=13.9
print(df.iloc[17,3])

#row:55 value:12.07 column:LOW 5
df.iloc[55,5]=12.0
print(df.iloc[55,5])

#row:61 value:16.24 column:HIGH 
df.iloc[61,3]=16.2
print(df.iloc[61,3])

#row:103 value:13.11 column:LOW 
df.iloc[103,5]=13.1
print(df.iloc[103,5])

df.isnull().values.any() 

"""Question2:"""

#Adding a line at the end of the dataframe 
df.loc[len(df.index)]=[np.nan, #MONTH 
                       np.nan, #DAY
                       df[["TEMP"]].mean(),#TEMP
                       df[["HIGH"]].max(), #HIGH
                       np.nan, #TIME
                       df[["LOW"]].min(), #LOW
                       np.nan, #TIME.1
                       df[["HDD"]].sum(), #HDD 
                       df[["CDD"]].sum(), #CDD 
                       df[["RAIN"]].sum(), #RAIN
                       np.nan, #W_SPEED
                       df[["WINDHIGH"]].min(), #WINDHIGH
                       np.nan, #TIME.2
                       np.nan] #DIR 

df.iloc[365]

"""Question3:"""

#median and standard deviation 
df[["TEMP"]].median() # -->16.8
df[["TEMP"]].std() #-->  7.405257

"""Question4:"""

#No of days the wind was blowing from each direction

pie=df.groupby(['DIR']).size().reset_index(name='counts')

#pie chart
dfpie=pd.DataFrame(pie)
dfpie.info()


plt.figure(figsize =(40, 25)) #size control of chart as otherwise % and labes do not show correctly
plt.figure(1)
plt.pie(dfpie['counts'], labels= dfpie['DIR'], shadow = False, autopct = '% 1.1f%%')

"""Question5:"""

#time max HIGH temperatures have occured, column time reflects time for HIGH temperatures
HIGHtime=df.groupby(['TIME']).size().reset_index(name='counts')
dfHIGHtime=pd.DataFrame(HIGHtime)
print(dfHIGHtime)

dfHIGHtime[["counts"]].max() 
print(dfHIGHtime[dfHIGHtime['counts']==23].index.values) #finds location of max value 
dfHIGHtime.iloc[34,0] #finds time value based on location of max value 
#result:datetime.time(15, 20)

#time max LOW temperatures have occured, column time1 reflects time for LOW temperatures
LOWtime=df.groupby(['TIME.1']).size().reset_index(name='counts')
dfLOWtime=pd.DataFrame(LOWtime)
print(dfLOWtime)

dfLOWtime[["counts"]].max() 
print(dfLOWtime[dfLOWtime['counts']==27].index.values) #finds location of max value 
dfLOWtime.iloc[0,0] #finds time value based on location of max value 
#result:datetime.time(0, 0)

"""Question6:"""

#day of year with max varience

df['HIGH'] = df['HIGH'].map(lambda x: float(x))
df['LOW'] = df['LOW'].map(lambda x: float(x))

diakimansi= pd.DataFrame([df.MONTH, df.DAY]).transpose() #creates new df with columns of old
diakimansi['temp_diff'] = df['HIGH'].sub(df['LOW'], axis = 0)
print(diakimansi) 


diakimansi.temp_diff[0:365].max() #last row is excluded as it is the dif. between max tem high and low
print(diakimansi[diakimansi['temp_diff']==15.599999999999998].index.values)
diakimansi.iloc[132,0] 
diakimansi.iloc[132,1] # result: MAY 13


"""Question7:"""

#From which direction did the wind blow the most days of the year 
wind=df.groupby(['DIR']).size().reset_index(name='counts')
print(wind)
wind[["counts"]].max() 
print(wind[wind['counts']==103].index.values)
wind.iloc[3,0] # DIR: N 

"""Question8:"""

#Direction of wind that gave the max windhigh

df['WINDHIGH'] = df['WINDHIGH'].map(lambda x: float(x))
windhigh=pd.DataFrame([df.DIR, df.WINDHIGH]).transpose() #creates new df with columns of old

windhigh.WINDHIGH[0:365].max() #last row is excluded as it is the dif. between max tem high and low
print(windhigh[windhigh['WINDHIGH']==64.4].index.values)
windhigh.iloc[89,0] # DIR: N

"""Question9:"""

#TEMP for every wind DIR and max, min with name of DIR
df['TEMP'] = df['TEMP'].map(lambda x: float(x))

DIR_TMP=df.groupby(by='DIR')['TEMP'].mean()
DIR_TMP_DF=pd.DataFrame(DIR_TMP) #creates dataframe of mean temp for every wind direction
DIR_TMP_DF.TEMP.max() # --> 21.03928571428571
DIR_TMP_DF.TEMP.min() # --> 11.864285714285714

print(DIR_TMP_DF[DIR_TMP_DF['TEMP']==21.03928571428571].index.values) #DIR:SW of max mean temp
print(DIR_TMP_DF[DIR_TMP_DF['TEMP']==11.864285714285714].index.values) #DIR:NW of min mean temp

"""Question10:"""

#Bar plot of rain per month

df['RAIN'] = df['RAIN'].map(lambda x: float(x))

MONTH_RAIN=df.groupby(by='MONTH')['RAIN'].sum()
print(MONTH_RAIN)

x=('JAN','FEB', 'MAR', 'APR', 'MAY', 'JOU', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
y=(94.2, 31.4, 110.8, 12.2, 68.2, 44.4, 11.8, 10.6, 15.4, 12.2, 142.4, 64.2 )
plt.figure(2)
plt.title('Rain per moth for 2017')
plt.bar(x,y)


"""Question11:"""

#Regression for DEC 2017 

seasons= pd.DataFrame([df.MONTH, df.DAY, df.TEMP]).transpose()

DEC_REG=seasons.query('MONTH == "DEC"')
DEC_REG_DF=pd.DataFrame(DEC_REG)
print(DEC_REG_DF)
DEC_REG_DF['TEMP'] = DEC_REG_DF['TEMP'].astype(float)
DEC_REG_DF['DAY'] = DEC_REG_DF['DAY'].astype(float)
DEC_REG_DF.info()

#creating table A which has as 1st column x and second 1
A = np.vstack([DEC_REG_DF.DAY, np.ones(len(DEC_REG_DF.DAY))]).T
print(A)

#Least-squares 
np.linalg.lstsq(A,DEC_REG_DF.TEMP,rcond='warn')
#b:14.08129032
#a:-0.18048387 so y=(-0.18048387)*x + (14.08129032)

#ploting the data (to make sure line is ok)
plt.figure(3)
_ = plt.plot(DEC_REG_DF.DAY, DEC_REG_DF.TEMP, 'o', label='Original data', markersize=10)
_ = plt.plot(DEC_REG_DF.DAY, (-0.18048387)*DEC_REG_DF.DAY + 14.08129032 , 'r', label='Fitted line')
_ = plt.legend()
plt.show()


#TEMP on 25/12/2018
#jan:31, Feb:28, March:31, April:30, May:31, Jun:30, Jul: 31,Aug: 31, Sept:30, Oct:31, Nov:30, Dec: 25
#so: on the 25/12/2018, 359 extra days will have passed... so 31+359 the value will be 390
print ((-0.18048387)*390 + (14.08129032)) #-56.30741898



"""Question12:"""

#4 graphs of winter, spring, summer and autumn with max,min and mean temp


seasons= pd.DataFrame([df.MONTH, df.DAY, df.TEMP]).transpose()

#winter
DEC=seasons.query('MONTH == "DEC" and DAY > 16')
JAN=seasons.query('MONTH == "JAN"')
FEB=seasons.query('MONTH == "FEB" and DAY < 18')

winter1=pd.DataFrame(DEC)
winter2=pd.DataFrame(JAN)
winter3=pd.DataFrame(FEB)

frames1 = [winter1, winter2, winter3]
result1 = pd.concat(frames1)

w1=result1.TEMP.max()# max: 14.4
w2=result1.TEMP.min()# min: -1.1
w3=result1.TEMP.mean()# mean: 7.715873015873015

#spring
MAR=seasons.query('MONTH == "MAR"')
APR=seasons.query('MONTH == "APR"')
MAY=seasons.query('MONTH == "MAY"')

spring1=pd.DataFrame(MAR)
spring2=pd.DataFrame(APR)
spring3=pd.DataFrame(MAY)

frames2 = [spring1, spring2, spring3]
result2 = pd.concat(frames2)

sp1=result2.TEMP.max()# max: 26.3
sp2=result2.TEMP.min()# min: 9.1
sp3=result2.TEMP.mean()# mean: 16.065217391304355

#summer
JUN=seasons.query('MONTH == "JOU"')
JUL=seasons.query('MONTH == "JUL"')
AUG=seasons.query('MONTH == "AUG"')

summer1=pd.DataFrame(JUN)
summer2=pd.DataFrame(JUL)
summer3=pd.DataFrame(AUG)

frames3 = [summer1, summer2, summer3]
result3 = pd.concat(frames3)

s1=result3.TEMP.max()# max: 35.9
s2=result3.TEMP.min()# min: 18.0
s3=result3.TEMP.mean()# mean: 26.69239130434783


#autumn
SEP=seasons.query('MONTH == "SEP"')
OCT=seasons.query('MONTH == "OCT"')
NOV=seasons.query('MONTH == "NOV"')

autumn1=pd.DataFrame(SEP)
autumn2=pd.DataFrame(OCT)
autumn3=pd.DataFrame(NOV)

frames4 = [autumn1, autumn2, autumn3]
result4 = pd.concat(frames4)

a1=result4.TEMP.max()# max: 29.2
a2=result4.TEMP.min()# min: 8.8
a3=result4.TEMP.mean()# mean: 18.2934065934066

 

#PLOT
plt.figure(4)
plt.subplot(1, 4, 1)
plt.plot(0,w1,'ro', 0, w2, 'bo', 0, w3,'go')
plt.xlabel('Winter')
plt.xticks([])
plt.ylim((-5,40))

plt.subplot(1, 4, 2)
plt.plot(0,sp1,'ro', 0, sp2, 'bo', 0, sp3, 'go') 
plt.xlabel('Spring')
plt.xticks([])
plt.ylim((-5,40))

plt.subplot(1, 4, 3)
plt.plot(0,s1,'ro', 0, s2, 'bo', 0, s3, 'go') 
plt.xlabel('Summer')
plt.xticks([])
plt.ylim((-5,40))

plt.subplot(1, 4, 4)
plt.plot(0,a1,'ro', 0, a2, 'bo', 0, a3, 'go') 
plt.xlabel('Autumn')
plt.xticks([])
plt.ylim((-5,40))

plt.tight_layout(pad=3.0)
plt.suptitle("Summary of Temperatures of 2017 per season")
plt.show()

"""Question13:"""

#Function regarding SUM of rain

def guess(users_input) :
    if users_input < 400: 
        print("Λειψυδρία")
    elif users_input >=400 and users_input <600: 
        print("Ικανοποιητικά ποσά βροχής")
    else:
        print("Υπερβολική βροχόπτωση")

#to call the function use: guess(users_input) or print(guess(users_input))







