import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygal.maps
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from imblearn.over_sampling import SMOTE
import country_converter as coco


df1 = pd.read_csv('articleInfo.csv')
df2 = pd.read_csv('authorInfo.csv')
df = pd.merge(df1, df2, on = 'Article No.', how = 'inner')
df.fillna(0, inplace=True)
pd.set_option('display.max_columns',None)


#Question 1
# plt.bar(df['Year'], df['Article No.'], width=0.5, color='b')
# plt.title("yearly publication")
# plt.xlabel('Year')
# plt.ylabel('Articles')
# plt.show()

#question 2
# plt.bar(df['Year'], df['Citation'], width=0.5, color='r')
# plt.title("yearly Citation")
# plt.xlabel('Year')
# plt.ylabel('Citation')
# plt.show()


#Question 3
cc = coco.CountryConverter()
Iso2_countries=coco.convert(names=df['Country'],to='ISO2',not_found='None')
print(Iso2_countries)
worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'number of Publications per country'
worldmap_chart.add('countries',Iso2_countries)

worldmap_chart.render()
worldmap_chart.render_to_file('map5.svg')

#question 4
# print("Question 1 Part.4")
# new_df=df.groupby(['Article No.','Author Affiliation']).count()
# new_df2=df.sort_values(['h-index','Author Name'], ascending=False)
# print(new_df)

#Question 5
# new_df2 = df.groupby(['h-index','Author Name']).count()
# print(new_df2)
# new_df = df.grousort_values(['Article No.'] ,ascending=False)


############################################Question 2###################################################################
# df=pd.read_csv('data.csv')
# df.head(60)
# df = df.fillna(0)
# df.isna().sum()
# del df[df.columns[-1]] #on database  that i recive, i got an extra column in blanck. for that reason, i use this command
# print(df.corr(method='pearson')['SUS'].sort_values())
# fig = plt.figure(figsize=(16,9))
# new_df = df.iloc[0:60]
# ax1 = fig.add_subplot(121)
# sns.distplot(new_df.loc[new_df['Purchase'] == 1]['SUS'], color='g')
# ax1.set_title('Distribution of SUS for every Ticket buy')
#
# ax2 = fig.add_subplot(122)
# sns.distplot(new_df.loc[new_df['Purchase'] == 0]['SUS'], color='r')
# ax2.set_title('Distribution of SUS for every Ticket not buy')
#
# plt.show()
# y= new_df['SUS']
# x = new_df.drop(columns='SUS')
# x = sm.add_constant(x)
#
# model = sm.OLS(y, x).fit()
# print(model.summary())
#
# N=new_df['SUS']
# M = new_df.drop(columns=['SUS','Gender','Duration'])
#
# M_train, M_test, N_train, N_test = train_test_split(M, N)
#
#
# lr = LinearRegression().fit(M_train,N_train)
#
# y_train_pred = lr.predict(M_train)
# y_test_pred = lr.predict(M_test)
#
# print("The R square score of linear regression model is: ", lr.score(M_test,N_test))

#############################################QUESTION 3#################################################################

#
# del df[df.columns[-1]]
# df.dropna(inplace= True)
#
#
#
# y = df['Purchase'].to_numpy()
# X= df.drop('Purchase', axis=1).to_numpy()
#
# scale = StandardScaler()
# scaled_X = scale.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.8)

# lc = LogisticRegression()
# svm = svm.SVC(probability=True)
# rfc = RandomForestClassifier()
# nbc = GaussianNB()
#
# lc.fit(X_train, y_train)
# svm.fit(X_train, y_train)
# rfc.fit(X_train, y_train)
# nbc.fit(X_train, y_train)
#
#
#
# y_lc_predicted = lc.predict(X_test)
# y_lc_pred_proba = lc.predict_proba(X_test)
#
# y_svc_predicted = svm.predict(X_test)
# y_svc_pred_proba = svm.predict_proba(X_test)
#
# y_nbc_predicted = nbc.predict(X_test)
# y_nbc_pred_proba = nbc.predict_proba(X_test)
#
# y_rfc_predicted = rfc.predict(X_test)
# y_rfc_pred_proba = rfc.predict_proba(X_test)

# print("######################MODEL WITHOUT applying oversample###############################")
# print("########Logistic Regression#########")
# print(classification_report(y_test, y_lc_predicted))
# print("########SVM#########")
# print(classification_report(y_test, y_svc_predicted))
# print("########Naive Bayes#########")
# print(classification_report(y_test, y_nbc_predicted))
# print("########Random Forest#########")
# print(classification_report(y_test, y_rfc_predicted))

# oversample = SMOTE()
# over_sampled_X_train, over_sampled_y_train = oversample.fit_resample(X_train, y_train)
#
# lc2 = LogisticRegression()
# svm2 = svm.SVC(probability=True)
# rfc2 = RandomForestClassifier()
# nbc2 = GaussianNB()
#
# lc2.fit(over_sampled_X_train, over_sampled_y_train)
# svm2.fit(over_sampled_X_train, over_sampled_y_train)
# rfc2.fit(over_sampled_X_train, over_sampled_y_train)
# nbc2.fit(over_sampled_X_train, over_sampled_y_train)
#
# y_lc2_predicted = lc2.predict(X_test)
# y_lc2_pred_proba = lc2.predict_proba(X_test)
#
# y_svc2_predicted = svm2.predict(X_test)
# y_svc2_pred_proba = svm2.predict_proba(X_test)
#
# y_nbc2_predicted = nbc2.predict(X_test)
# y_nbc2_pred_proba = nbc2.predict_proba(X_test)
#
# y_rfc2_predicted = rfc2.predict(X_test)
# y_rfc2_pred_proba = rfc2.predict_proba(X_test)
#
# print("######################################MODEL applying oversample################################")
# print("########Logistic Regression#########")
# print(classification_report(y_test, y_lc2_predicted))
# print("########SVM#########")
# print(classification_report(y_test, y_svc2_predicted))
# print("########Naive Bayes#########")
# print(classification_report(y_test, y_nbc2_predicted))
# print("########Random Forest#########")
# print(classification_report(y_test, y_rfc2_predicted))