# <a name="top"></a>Sngle Family Home Value Prediction
![]()

by: Victoire Migashane

<p>
  <a href="https://github.com/MigashaneVictoire" target="_blank">
    <img alt="Victoire" src="https://img.shields.io/github/followers/MigashaneVictoire?label=Follow_Victoire&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

<!-- <img src="https://docs.google.com/drawings/d/e/2PACX-1vR19fsVfxHvzjrp0kSMlzHlmyU0oeTTAcnTUT9dNe4wAEXv_2WJNViUa9qzjkvcpvkFeUCyatccINde/pub?w=1389&amp;h=410"> -->

## <a name="project_description"></a>Project Description:
[[Back to top](#top)]

**Scenario**

You are a junior data scientist on the Zillow data science team and receive the following email in your inbox:

We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.

We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

One last thing, Maggie lost the email that told us where these properties were located. Ugh, Maggie :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

**Business Goals**

- Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

- Find the key drivers of property value for single family properties. Some questions that come to mind are:

- Why do some properties have a much higher value than others when they are located so close to each other?
Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?
- Is having 1 bathroom worse for property value than having 2 bedrooms?
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

- Make recommendations on what works or doesn't work in predicting these homes' values.

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:

- Create all the files I will need to make a functioning project (.py and .ipynd files)
- Create a .gitignore file and ignore my env.py file
- Start by acquiring data from the Codeup database and document all my initial acquisition steps in the acquire.py file
- Using the prepared file, clean the data and split it into the train, validate, and test sets.
- Explore and analyze the data. (Focus on the main questions)
    - Why do some properties have a much higher value than others when they are located so close to each other?
    - Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?

- Answer all the questions with statistical testing.
    - Identify drivers of home value.
- Predict tax value using driving features.
- Document findings (include 4 visuals)
    - Add important findings to the final notebook
    - Create a csv file of test predictions on the best perfomring model.



### Questions

1. Is there a linear relationship between the square footage of the property and the tax value?
    - Does the mean tax value differ significantly between properties with different square footage?

2. Do the number of bedrooms or bathrooms impact the tax value?
    - Is there a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms
    
3. Does the county where the property is located affect the tax value?
    - Does the mean tax value differ significantly between properties in Los Angeles County compared to the other counties?

- 4. Does the year the property was built have any influence on its tax value?
    - Is there a significant difference in the mean tax value between properties built in different years?

### Target variable

- The target variable is the `Tax_value` column in the data.

### Deliverables:

A. Github repo with:

- a complete readme.md
- acquire module (.py)
- prepare module (.py)
- a final report (.ipynb)
- other supplemental artifacts created while working on the project (e.g. exploratory/modeling notebook(s)).

### With more time:

- 

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]




***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Column Name | Description |
| ----------- | ----------- |
| bedrooms | The number of bedrooms in the property. Bedrooms refer to individual rooms used primarily for sleeping and are typically found in residential properties. |
| bathrooms | The number of bathrooms in the property. Bathrooms refer to rooms containing a toilet, sink, and typically a bathtub or shower, used for personal hygiene. |
| sqr_feet | The total square footage of the property. Square footage is a measurement of the area covered by the property, indicating its size or living space. It is often used to estimate the property's value or to determine the price per square foot. |
| tax_value | The assessed value of the property for tax purposes. Tax value represents the estimated worth of the property as determined by the local tax authority. It is used to calculate property taxes. |
| year_built | The year in which the property was constructed or built. This indicates the age of the property and can be useful in assessing its condition or historical significance. |
| tax_amount | The amount of tax owed on the property. Tax amount refers to the actual dollar amount that needs to be paid in property taxes based on the assessed tax value and local tax rates. |
| county | The county where the property is located. County refers to a specific geographic region or administrative division within a state or country. It helps identify the property's location within a broader jurisdiction. |

***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Wrangle steps: 

I am using the Telco churn data from the Codeup database.

- Query the following columns:
    - `bedroomcnt, bathroomcnt,calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips`

- 2152863 rows and 7 columns.
- 7 numeric and 0 object
- 22778 total null count (1% of the data)
- Remove all nulls (1% of the data)
- Remove duplicated rows.
- convert data type from float to int (bedrooms, bathrooms)
- remove outliers
- replace the fips code with county names and Encode county column.
- Split data into train, validate, and test. (`60/20/20 ` split)
- scale the numeric categorical and continuous variables and extract a copy of the original data frame.
    - `bedrooms, bathrooms, sqr_feet, year_built,`

*********************

## <a name="explore"></a>Data Exploration amd Statistical Analysis
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py 
    - explore.py
    - modeling.py


### Takeaways from exploration and statistical analysis
- The test says that the mean tax value does not differ significantly between properties with different square footage.

- The test says that there is a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms.

- Test show that the mean tax value does not differ significantly between properties in Los Angeles County compared to the other counties.

- We have data to say that there is not a significant difference in the mean tax value between properties built in different years.

***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:

### Baseline
    
- mean RMSE basline: 288337.85
- median RMSE baseline: 296095.82

- `bedroomcnt, bathroomcnt,calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, county`

***

### Models and R<sup>2</sup> Values:
- Will run the following regression models:

    

- Other indicators of model performance with breif defiition and why it's important:

    
    
#### Model 1: Linear Regression (OLS)

OLS Regressor 
- Train RMSE: `235399.31`
- Validate RMSE: `235138.62`
- RMSE_difference `-260.69`
- R2_validate `0.33`

### Model 2 : Lasso Lars Model

Lasso + Lars
- Train RMSE: `235399.22` 
- Validate RMSE:  `235138.92`
- RMSE Difference:  `-260.30`
- R2_validate: `0.33`

### Model 3 : Tweedie Regressor (GLM)

Tweedie Regressor

- Train RMSE: `236386.04` 
- Validate RMSE:  `236127.92`
- RMSE Difference:  `-258.11`
- R2_validate: `0.33`

### Model 4: Polynomial Regression Model

Polynomial feature regressor (Power 3)

- Train RMSE: `229654.86` 
- Validate RMSE:  `229364.37`
- RMSE Difference:  `-290.49`
- R2_validate: `0.37`


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- |
| Baseline | 288337.85 ||
| Linear Regression (OLS) | 235138.62 |0.33|  
| Tweedie Regressor (GLM) | 236127.92 |0.33|  
| Lasso Lars | 235138.92 |0.33|  
| Cubic Regression | 229364.37 |0.37| 
- {} model performed the best


## Testing the Model

- Cubic Regression

***

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]

## Conclusion

### Explore

- The test says that the mean tax value does not differ significantly between properties with different square footage.

- The test says that there is a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms.

- Test show that the mean tax value does not differ significantly between properties in Los Angeles County compared to the other counties.

- We have data to say that there is not a significant difference in the mean tax value between properties built in different years.

### Modeling

The final (Cubic Regression) model performed significantly better than the baseline by about 5%. Saving $100000 since 1% is about $20000.

- Bedrooms, bathrooms, square feet, year built, and county seems to be moderate drivers for predicting tax value.

### Recommendations

Some properties have a much higher value than others when they are located so close to each other mostly due to the size, year built, and the number of rooms in the house. So to better predict home value we should look father into the condition of the house and identify a unique indoor or outdoor feature that some homes have that makes them worth more or less to the current housing market.

### Next Steps

- Collect bettwe data on current home conditions.

