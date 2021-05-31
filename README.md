---
layout: post
title: Time Series Forecasting with TensorFlow, ARIMA, and PROPHET (4-min read)
comments: true
---
I have been preparing weekly for the <a href="https://www.tensorflow.org/certificate" target="_blank">TensorFlow Developer Certificate</a> by taking a deep dive into an individual deep learning concept and exploring the TensorFlow applications.  This week we'll dive into Time Series Forecasting, and extremely powerful approach to predicting the future.  It has many useful applications and is a very common strategy in the retail space as well as weather or production forecasting and even used by NASA searching for earth-like planets! 

This project explored the fundamentals of time series analysis and forecasting starting with a robust weather dataset to be used in multivariate analysis as well as a superstore retail sales dataset with advanced forecasting tools.  Multivariate regression is an extremely powerful approach and can be applied to a variety of industries including oil and gas well performance prediction.

<p align="center">
    <img src="../images/time_series.jpg" id="tsa" alt="Time Series Forecasting">
</p>

A generalized example of forecasting is shown above, but the concept is fairly straightforward.  Allow a sophisticated deep learning network to learn the ebbs and flows of a time series of data (weather, stock performance, sales, etc.) based on various features and use these learnings to project into the future.  The simplest form of this type of problem is a linear regression, or a best-fit line where you essentially "eye-ball" or average out the trend of the data and continue along that line.  Advanced, deep learning multivariate regressions can learn very nuanced patterns in the data including seasonality, day/night cycle, weekends, holidays, etc. to achieve very impressivep performance.

## Multivariate Regression with TensorFlow ##

I encourage you to take a deeper look in the notebook hosted on GitHub if interested in the details of the individual steps, but there are multiple approaches to forecasting.  To begin building up our understanding of the capabilities, simple models were created to predict the values just one time-step into the future.  Although this can have useful applications, it is much more powerful to be able to "see" further into the future where the various dynamic features all impact the results which we will dive into later.

Through a process called data windowing, we segmented our hourly weather data into 24-hour day intervals which are fed into the model for training.  As we built up more and more sophisticated models the ability to identify the trends of the training data improved.  Similarly to the NLP problems we worked on previously, the Long Short-Term Memory or LSTM model is a very powerful algorithm for forecasting as it is able to pass information through the model to better understand the nuances of the sequencing.  Similarly to NLP where the context of a word carries significance, in a time series problem where the features tend to move together over time that context provides increased learning and performance.  An example shown below. 

<p align="center">
    <img src="../images/lstm_forecast.jpg" id="lstm" alt="LSTM Forecast Example">
</p>

That is a good looking model.  As we can see, it fits nice and snug with the training data.  Evaluating a regression model is a little less straightforward than a classification problem.  There are various metrics, and one commonly used example is the Mean Absolute Error which is a process of finding the error of each prediction and averaging them.  In this case, a lower number is better meaning there is less error in the predictions.  The LSTM model achieved an MAE of 0.0527, an improvement from the 0.0852 baseline!

As mentioned previously, typically we are more interested in accurately predicting results much further into the future.  It hardly needs an explanation why this type of information would be beneficial to a business, person, government, etc. and has become a very popular avenue of deep learning.  The underlying foundation is carried over however the output of the models is modified.  There are two distinct approaches for forecasting, either making all of the predictions at once in the **single-shot** method, or progressively step-by-step using each prediction as input for the next in what is called **autoregression**.  They both have their benefits, and like always there is no "best" it will always depend on the situation.  Single-shot methods are a little easier to understand conceptually, the model learns the pattern and will make an output for a given length.  The autoregression method is shown in an illustration below.

<p align="center">
    <img src="../images/autoregression.jpg" id="ar" alt="Autoregression Summary">
</p>

Definitely a bit more complex, and more complex does not always lead to higher performance.  But it is a very powerful approach that logically makes sense.  As we know, things like weather or stock prices will generally change slowly over time.  It really is more a challenge of predicting the change from one day to the next, moreso than predicting the actual value.  Autoregression models can be created to learn in this fashion as well.

<p align="center">
    <img src="../images/ar_lstm.jpg" id="ar-lstm" alt="Autoregression LSTM Forecast">
</p>

As mentioned previously, the LSTM lends itself very well to time series problems.  We used the LSTM model to implement the autoregression algorithm to compare performance.  As you can see in the forecast above, the model is performing decently well but it is a challenge the further you get from the training data.  Comparing the Autoregression LSTM to the single-shot version, it actually performed slightly worse.  In this case, the added complexity of the model ended up going beyond the point of diminishing returns.  It is always worth exploring various architectures, however to compare.

<p align="center">
    <img src="../images/results.jpg" id="results" alt="Forecast MAE Results">
</p>

## Sales Forecasting with ARIMA and Prophet ##

Switching gears now, I also wanted to take some time to explore two of the most powerful forecasting tools used today ARIMA and Prophet.  ARIMA stands for Autoregressive Integrated Moving Average.  As the name implies, this is a powerful autoregression technique that also factors in the moving average of the time series.  It can also be adapted to problems with seasonality by using the SARIMA model, and either can be upgraded to a multiple output model by using ARIMAX or SARIMAX.

Additionally, the team at Facebook developed an extremely powerful tool aptly named the Prophet, as it can see far into the future.  The prophet model has become very common as it is very customizable, powerful, and there are additional visualization capabilities that we will not explore at this time.  Let's take a look at both of them now using the retail sales dataset.

<p align="center">
    <img src="../images/sales_decomposition.jpg" id="decomposition" alt="Sales Data Decomposition">
</p>

The ARIMA and Prophet models both require extensive, custom preprocessing to function properly.  After performing a lot of great EDA and preprocessing that you can enjoy at the GitHub repo, we got into it.  Taking a look at the sales data decomposition above, we can see there is plenty of noise and seasonality in this data which makes it difficult to accurately forecast.  But that is where SARIMAX and PROPHET will help us.  After analyzing the data to identify the proper parameters for the model, the fitting and forecasting process was relatively straightforward.  SARIMAX forecast is shown below.

<p align="center">
    <img src="../images/sarimax_forecast.jpg" id="sarimax" alt="SARIMAX Sales Forecast">
</p>

Taking a look at the results, the orange line is the predicted sales and the grey window represents the 95% confidence interval.  As we can see, the prediction is very nuanced in the first 12-18 months where it still feels more confident, however from there it simply repeats the same pattern including a slight positive trend, but with less and less confidence over time.  This is expected, as it is harder to predict further into the future.  These sales forecasts can also be easily output as a file to be sent to interested stakeholders or used for various analyses and visualizations.

Let's take a look at the Prophet model.  It is a more advanced model and is able to learn even more details of the patterns and trends.  It is able to predict with much higher confidence than the SARIMAX model achieved.  Looking at total sales, we can see the very detailed seasonality as well as the overall positive trend.

<p align="center">
    <img src="../images/prophet_forecast.jpg" id="prophet" alt="Prophet Sales Forecast">
</p>

This has been a great introduction to Time Series Analysis and Forecasting.  We went through a detailed TensorFlow.org multivariate analysis walkthrough to get our bearings before applying the concepts to a real world problem with retail sales data.  Time Series Forecasting is extremely powerful, and the actionable intelligence can be applied to any industry to help drive business decisions and optimize operations.  This has been great, looking forward to what's next!

If you would like to dive into the code please check out the <a href="https://github.com/polzinben/Time-Series-Forecasting" target="_blank">GitHub repo.</a>

Resources:
- https://www.tensorflow.org/tutorials/structured_data/time_series
- https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
- https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/
