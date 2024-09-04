import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#used template idea from exercises
OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_high_unemployment_normality_p:.3g} {initial_low_unemployment_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_high_unemployment_normality:.3g} {transformed_low_unemployment_normality:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "T-test p-value after data transformation: {ttest_p:.3g}\n"
    "Mann_whitneyu_test_pvalue: {mann_whitneyu_test:.3g}\n"
    "Chi2_contingency_p: {chi2_contingency_p:.3g}"
)

def examine_stats(data_aggregate):
    #the scatterplot shows the countries which are outliers in Population
    #removing those to get better stats
    data_aggregate = data_aggregate[(data_aggregate['Country Name'] != 'United States')]
    data_aggregate = data_aggregate[(data_aggregate['Country Name'] != 'Italy')]

    #plt.boxplot(data["Unemployment rate"]) 

    #categorize data into low unemployment and high unemployment rate
    high_unemployment = data_aggregate[data_aggregate["Unemployment rate"] >= 7]
    low_unemployment = data_aggregate[data_aggregate["Unemployment rate"] < 7]

    #check if average poulation per country has an effect on uemployment rate
    #Null Hypothesis average population of countries with high unemployment is equal to average population of countries with low unemployment
    #Does Average population of countries has effect on the employment rate?
    ttest = stats.ttest_ind(high_unemployment['population'],low_unemployment['population'])
    ttest_pvalue = ttest.pvalue #positive p cannot reject the null hypothesis

    #Plot a histogram
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.hist([high_unemployment['population'],low_unemployment['population']])
    plt.title(" Original - Population Distribution")
    plt.legend(['High Unemployment','Low Unemployment'])
    plt.xlabel('Population in Thousands')
    plt.ylabel('Unemployment rate')


    #Normality Test
    high_unemployment_normality_p = stats.normaltest(high_unemployment['population']).pvalue
    low_unemployment_normality_p = stats.normaltest(low_unemployment['population']).pvalue

    employment_levene_p = stats.levene(high_unemployment['population'],low_unemployment['population']).pvalue

    #trasforming data using log as it is not normally distributed
    high_unemployment_log = np.log(high_unemployment['population'])
    low_unemployment_log =  np.log(low_unemployment['population'])

    #plot a histogram after transforamtion
    plt.subplot(1, 2, 2)
    plt.hist([high_unemployment_log,low_unemployment_log])
    plt.title("Transformed - Population Distribution")
    plt.legend(['High Unemployment after Log','Low Unemployment after Log'])
    plt.xlabel('Population in Thousands')
    plt.ylabel('Unemployment rate')
    
    #savefig
    plt.savefig('Histogram_unemployment.png')

    #normality test after transforamtion
    high_unemployment_normality_log_p = stats.normaltest(high_unemployment_log).pvalue
    low_unemployment_normality_log_p = stats.normaltest(low_unemployment_log).pvalue

    #levene test (equal variance test) after transformation
    employment_levene_log_p = stats.levene(high_unemployment_log,low_unemployment_log).pvalue

    #ttest after transformation
    ttest_log = stats.ttest_ind(high_unemployment_log,low_unemployment_log)
    ttest_pvalue_log = ttest_log.pvalue
    
    #again we are unable to conclude as p > 0.05
    #Data does not pass the normality test
    #performing non parametric test
    p_employment_population= stats.mannwhitneyu(high_unemployment['population'],low_unemployment['population'], alternative="two-sided").pvalue
    
    #chi test
    #dividing data into 4 categories
    high_unemployment_high_population = len(high_unemployment[high_unemployment['population'] >= 40])
    high_unemployment_low_population = len(high_unemployment[high_unemployment['population'] < 40])
    low_unemployment_high_population = len(low_unemployment[low_unemployment['population'] >= 40])
    low_unemployment_low_population = len(low_unemployment[low_unemployment['population'] < 40])
    
    contingency = [[high_unemployment_high_population, high_unemployment_low_population], [low_unemployment_high_population, low_unemployment_low_population]]
    
    res = stats.chi2_contingency(contingency)
    p_unemployment = res.pvalue
    
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=ttest_pvalue,
        initial_high_unemployment_normality_p=high_unemployment_normality_p,
        initial_low_unemployment_normality_p=low_unemployment_normality_p,
        initial_levene_p= employment_levene_p,
        transformed_high_unemployment_normality= high_unemployment_normality_log_p,
        transformed_low_unemployment_normality=  low_unemployment_normality_log_p,
        transformed_levene_p=employment_levene_log_p,
        ttest_p=ttest_pvalue_log,
        mann_whitneyu_test=p_employment_population,
        chi2_contingency_p = p_unemployment
    ))

    

def main():
   data = pd.read_csv('Data/merged_data.csv')
    
   #group the data by Country Name, Year, Unemployment rate
   data = data.groupby(['Country Name', 'Year','Unemployment rate']).agg({"population":"sum"})
   data = data.reset_index()

   # as the data is not structered and it is not available for each country for each year
   # so found the avergae population for each country 
   data_aggregate = data.copy()
   data_aggregate = data.groupby(['Country Name']).agg({"population":"mean","Unemployment rate":"mean"})
   
   data_aggregate['population'] = data_aggregate['population']/1000 #converting population into thousands
   data_aggregate['Unemployment rate'] = data_aggregate['Unemployment rate'].round(2)

   # Print aggregated stats for dataset
   print("Min stats")
   print(data_aggregate.idxmin())

   print("Max stats")
   print(data_aggregate.idxmax())

   data_aggregate = data_aggregate.reset_index()
   plt.scatter(data_aggregate['population'],data_aggregate['Unemployment rate'])
   plt.xlabel('Population in Thousands')
   plt.ylabel('Unemployment rate')

   plt.savefig('Population Vs Unemployment.png')
   
   examine_stats(data_aggregate)
   

if __name__ == '__main__':
    main()
