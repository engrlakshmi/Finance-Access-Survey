# Finance-Access-Survey
The means to access finance have seen a seismic shift in recent years. Traditional touchpoints like ATMs and bank branches are witnessing a decline, while non-traditional platforms such as retail agents and mobile money agents are surging. 
#The objective of the present study is to conduct an in-depth analysis of mobile money and digital banking adoption across various regions to understand disparities and identify areas for improvement.
##Source Data link - (https://data.imf.org/?sk=E5DCAB7E-A5CA-4892-A6EA-598B5463A34C)https://data.imf.org/?sk=E5DCAB7E-A5CA-4892-A6EA-598B5463A34C (The "Finance Access Survey" is likely a part of their data collection efforts to assess and monitor financial access and inclusion in various countries). 

A general outline of the steps involved are listed below:

1.**Data Collection and Preprocessing**:

Collect financial access survey data, including variables related to financial inclusion, economic indicators, and other relevant information.
Preprocess the data, which may involve cleaning, handling missing values, and encoding categorical variables.

2.**Data Splitting**:

Split the dataset into training and testing subsets. Typically, you might use 70-80% of the data for training and the remaining 20-30% for testing.

3. **Feature Selection**:

Identify the most important features or variables that are likely to impact financial access. Random Forest can handle a large number of features, but feature selection can improve model efficiency.

4. **Model Building**:

Train a Random Forest regression model on the training dataset. The Random Forest algorithm builds an ensemble of decision trees, which collectively provide predictions.
Hyperparameter Tuning:

Fine-tune the hyperparameters of the Random Forest model. Common hyperparameters include the number of trees, the maximum depth of the trees, and the number of features to consider at each split.

5. **Model Evaluation**:

Evaluate the model's performance on the testing dataset using appropriate evaluation metrics for regression tasks. Common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R^2).

6. **Feature Importance Analysis**:

Analyze feature importance scores generated by the Random Forest model to understand which variables have the most impact on financial access.

7. **Visualization**:

Create visualizations, such as feature importance plots or partial dependence plots, to explain the model's predictions and insights from the data.
Interpretation and Insights:

Interpret the model's results to gain insights into the factors influencing financial access. This step involves understanding the relationships between variables and their effects.

8. **Model Deployment**: 

If the model provides valuable insights, consider deploying it for ongoing monitoring and decision-making. This model is deployed using Flask API.



###Insights


1. **Digital Financial Services Are on the Rise**: The shift towards non-traditional platforms like mobile money and internet banking signifies a significant increase in the use of digital financial services. This trend is a response to the convenience and accessibility offered by these digital channels.

2. **Regional Variation in Preferences**: The choice between mobile money and internet banking varies by region. In Africa, mobile money is a preferred choice, while in Europe and the Western Hemisphere, internet banking is gaining popularity. These preferences may be influenced by factors like technology infrastructure and consumer behavior.

3. **Increased Transaction Volume**: The rise in digital financial service usage is reflected in the increased number and volume of transactions. This indicates a growing acceptance and trust in digital platforms for conducting financial activities.

4. **Digital Financial Services and Economic Growth**: In regions like Africa, the growth of mobile money transactions as a percentage of GDP highlights the role of digital finance in driving economic activity. It can foster financial inclusion and stimulate economic growth.

5. **Challenges for Traditional Banking**: Traditional banking, represented by ATMs and bank branches, is facing a decline in usage. This trend challenges banks to adapt and invest in digital infrastructure to remain relevant and competitive.

6. **Loan and Deposit Trends**: While there is an increase in the number of bank accounts globally, there's a discernible decrease in the value of outstanding deposits and loans relative to GDP. This suggests that people might be using their bank accounts for different purposes or exploring alternative means of accessing credit.

7.**Economic Policy Impact**: The decrease in loan values in some regions can be attributed to policy measures related to the COVID-19 pandemic. As these measures unwind and monetary policy tightens, there is an impact on bank lending and borrowing behavior.







