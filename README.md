## Master's Thesis: Predicting US Recessions with Machine Learning

### Read the Report 👉 [link](https://drive.google.com/file/d/1xIuXDUn-J9_CYk0OdG4s8mzdy6GQNIZm/view?usp=share_link)

<h3 style="text-align: center;">Abstract</h3>

This paper conducts a comprehensive horse race between several popular machine learning models and compares their performances with a traditional logistic regression model at predicting US recessions. Each model is tested at three-, six-, nine-, 12-, and 18-month horizons. A gradient boosting, random forest, support vector classifier, feedforward neural network, and custom consensus model are all compared against the logistic regression model. A dataset spanning 1962-2023 using six leading indicators is used for training and testing the models. The predictive ability of each model is evaluated using two scoring methodologies. The first scoring methodology is a novel ranking system using accuracy, precision, recall, F1, and AUROC scores. The second scoring methodology uses bootstrapped AUROC scores to perform a one-sided test on every model pair across each horizon. The results strongly suggest the gradient boosting, random forest and consensus models are generally superior at recession classification across horizons in comparison to the logistic regression model. The evidence for outperforming the logistic regression model is significantly weaker for the neural network and support vector classifier models. Finally, based on the findings of the consensus model, the results indicate combining the predictions of multiple models can produce better recession predictions than relying on the outputs of a single model.

<h3 style="text-align: center;">Why should we try to predict recessions with machine learning?</h3>

When looking at econometric data, the people behind those numbers are sometimes forgotten. There are hopes and dreams behind those numbers. According to Roelfs et al. (2011), losing a job can lead to 63% higher risk of mortality. Despite this, it can often be difficult to make the connection between something like the yield curve inverting and a family losing the home they saved for years to afford. In the grand scheme of the economy, these events are definitely connected.

The implications of better forecasting affect real people in a positive way, so it is imperative to find the best possible way to model recessions. Additionally, policy makers and business leaders base their own models on the work done by researchers on recession predictions and forecasting more generally. That is why it is important for researchers to critically and objectively evaluate different recession forecasting techniques. Critical decisions filter down from the findings of such research and it therefore matters a great deal which models are most adept at predicting recessions. This is why research into recession forecasting is important.

Naturally, if it were easy to predict recessions, there would be no need for this paper. Recessions often share similar characteristics but are usually caused by exogeneous shocks, which are difficult or impossible to foresee. For example, the 2001 recession in the US was caused by the popping of the tech-bubble and the tragic 9/11 attacks (Kliesen et al., 2003). On the other hand, the Great Recession of 2007-2009 was largely triggered by a collapse in the US housing market (Verick & Islam, 2010). Despite these vastly different causes, various leading indicators can still give reliable signals of future economic conditions (Mostaghimi, 2006). Therefore, while every recession is different, there is enough consistency in various leading indicators that suggest predicting future economic conditions is somewhat feasible. With that said, there is no silver-bullet indicator for predicting recessions.

Until recently, the modelling techniques used to identify relationships between eco- nomic indicators and recessions have remained very similar. Before the widespread adop- tion of machine learning models, the primary methodologies for research on recession prediction have been logistic and probit regression models. However, many factors such as an exponential increase in the amount of econometric data and a lower cost of computing are challenging this status quo in favor of more complex machine learning models (Sarker, 2021).

There is now growing evidence that suggests machine learning models are superior at recession forecasting than traditional logistic and probit models. However, it is still premature to say that there exists a definitive consensus that machine learning models are better at recession prediction than traditional econometric techniques. One reason for this are the differing scoring methodologies.
There are several common ways to compare binary classification models like the ones tested in this paper. Some popular scoring methodologies that are currently used to compare model performance are metrics such as the F1-score or performing significance tests using receiver operating characteristic (ROC) curves. Some commonalities in the scoring methodologies exist across research, though there is no standardized approach, and the results vary widely in part because of this.

This paper contributes to the research into recession prediction by performing a comprehensive horse race between several machine learning models to gauge general performance against a traditional logistic regression model. To this end, a gradient boosting, neural network, support vector classifier, random forest, and consensus model are all tested against a logistic regression model. The models’ classification abilities are evaluated at three-, six-, nine-, 12-, and 18-month time horizons. The models are then evaluated using two different scoring methodologies. The first scoring methodology is a novel ranking approach based on several different metrics collected across horizons. The second evaluation methodology uses bootstrapped area under the receiver operating characteristic (AUROC) scores to calculate p-values and perform one-sided tests between models.

The approach proposed here aims to fill a critical gap in existing research, that tends to focus disproportionately on specific forecasting horizons and inconsistent evaluation techniques. Consequently, many studies lack a comprehensive assessment of the overall performance capabilities of machine learning models relative to logistic regression models at recession predictions. The novel score ranking and bootstrapped AUROC methodology employed in this study directly address these issues by giving clear indications of general performance by using both common scoring metrics and significance tests.
