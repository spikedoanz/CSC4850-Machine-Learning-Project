# Predicting belief in astrology using OKCupid dating profile metadata
---

## Table Of Contents
1. [Project Summary](#project-summary)  
2. [Expectations](#expectations)
2. [Pipeline Info](#pipeline-info)
4. [Dataset and Features](#dataset-and-features)
4. [Technologies](#technologies)
3. [Machine Learning Models](https://github.com/manhtiendoan/CSC4850-Machine-Learning-Project/edit/main/README.md#machine-learning-models)
4. [Results](#results)
7. [Citations](#citations)  
8. [Special Thanks](#special-thanks)    
9. [Contributors](#contributors)  

## Project Summary  
>> This project involves the assessment of twelve machine learning models using the 2012 OkCupid dataset. This study aims to evaluate the performance of these models and determine which is most effective at classifying a user’s ‘Star Sign Intensity’. The performance of each model was evaluated based on metrics including accuracy, precision, recall, F1 score, and associated learning curves. Model selection, for each algorithm, compared three independent train/test splits (50-50, 70-30, and 80-20) before undergoing 10-fold cross-validation. The results of which were compared and the best models (by metrics) for each were selected by hand. The findings of this study do not necessarily support much in the way of predicting a human’s interest in star signs in the given context but do provide valuable insights into the appropriate selection of machine learning models and algorithms for any application.

## Expectations  
### Pipeline Info  
* Model initialization: All models were initialized with random_state = 1234 for reproducibility whenever possible  
* Data splits: The models were trained on 3 splits of the dataset in 3 ratios (50-50, 70-30, 80-20)  
* Cross validation: For every split, every model is trained using 10-fold cross validation of the training set, from which the best model is selected.  
* For classification models, we primarily used accuracy as the determining metric as our dataset is largely evenly split (47-53).  
* From the best models chosen for every split, we choose the best model for every model type.  
* Finally, from all the models we choose a single best performer.  
* Notes on model evaluation: models that perform close to or worse than 0.53 (always guessing a single class) will be classified as poorly performing.  

## Dataset and Features
### About the Dataset  
Dataset obtained from [Kaggle.com](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles)
>> OkCupid is a mobile dating app. It sets itself apart from other dating apps by making use of a precomputed compatibility score, calculated by optional questions the users may choose to answer.  In this dataset, there are 60k records containing structured information such as age, sex, orientation as well as text data from open ended descriptions.

### Raw Data  
* age, status, sex, orientation, body_type, diet, drinks, drugs, education, ethnicity, height, income, job, last_online, location, offspring, pets, religion, sign, smokes, speaks, essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9
* 59,949 raw entries 
* .csv format

For training and predicting, all used features were converted to numeric or binary data. These features are labeled with '_data' for use as an official for use in training and testing the models. 

### Feature Selection
 The following fields were used for classification: 
 >> 'age', 'height', 'income', 'sign_data', 'religion_data', 'religion_intensity', 'status_data', 'sex_data', 'height_data', 'orientation_data', 'body_type_data', 'diet_data' 'drinks_data', 'drugs_data', 'education_data', 'job_data', 'last_online_data', 'offspring_data', 'smokes_data', 'speaks_data', 'essay0_data', 'essay1_data', 'essay2_data', 'essay3_data', 'essay4_data', 'essay5_data', 'essay6_data', 'essay7_data', 'essay8_data', 'essay9_data', 'essay_len'.

### Star Sign Intensity   
Since predicting a persons Astrological Sign was not a solvable problem with this data set and these techniques an alternative metric was used: Star Sign Intensity. Star Sign Intensity is a composite feature based on self-reported OkCupid survey data representing one’s affinity or interest in their zodiac sign. This data was reported as part of the original column but separated by a comma from the original sign value. For example, an entry might contain "Leo, and it matters a lot'. The three possible options for this sub-field were combined into two to create a binary classification problem: "My sign matters" and "My sign doesn't matter".  

## Technologies  
### [Python](https://www.python.org/) <img src="https://user-images.githubusercontent.com/60898339/222571123-81f8e8e4-b183-4f92-a4bc-95d9d3e9f007.png" width=25 height=25>

### [Google Colab](https://colab.research.google.com/) <img src="https://user-images.githubusercontent.com/60898339/233802082-d2c46791-530f-4c95-9bd0-0b0889f8a601.png" width=25 height=25>

## Libraries
### [scikit-learn](https://scikit-learn.org/) <img src="https://user-images.githubusercontent.com/60898339/233802426-495b6620-22ba-4910-a63c-fec3d4843210.png" width=5% height=5%>
### [NumPy](https://numpy.org/) <img src="https://user-images.githubusercontent.com/60898339/233802193-1a22a918-5a56-4e45-8f09-77f58d65629d.svg" width=25 height=25>
### [Pandas](https://pandas.pydata.org/) <img src="https://user-images.githubusercontent.com/60898339/233802257-a731902d-9557-4707-bfae-2ea0dfb3bf4b.svg" width=55 height=35>
### [Matplotlib](https://matplotlib.org/) <img src="https://user-images.githubusercontent.com/60898339/233802324-53ef5e2f-c190-43b1-a763-6c889f8d87cb.svg" width=65 height=45>

## Machine Learning Models 
* [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Support Vector Machine (Linear Kernel)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
* [Support Vector Machine (RBF Kernel)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
* [Multi Layer Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* [Naive Bayes Classification](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
* [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)


## Results 
### Tables of models selected from 10-fold cross validation for each split.
<table width="0"  align="">
<tbody>
<tr>
<td width="139">
<p><strong>Split 0 (50/50)</strong></p>
</td>
<td width="139">
<p><strong>Accuracy</strong></p>
</td>
<td width="139">
<p><strong>Precision</strong></p>
</td>
<td width="139">
<p><strong>Recall</strong></p>
</td>
<td width="139">
<p><strong>F1-Score</strong></p>
</td>
</tr>
<tr>
<td width="139">
<p>Decision Tree</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>Perceptron</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.48</p>
</td>
</tr>
<tr>
<td width="139">
<p>Naive Bayes</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.48</p>
</td>
</tr>
<tr>
<td width="139">
<p>Logistic Regression</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM - Linear Kernel</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM - RBF Kernel</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.47</p>
</td>
</tr>
<tr>
<td width="139">
<p>Multilayer Perceptron</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.54</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.51</p>
</td>
</tr>
<tr>
<td width="139">
<p>Gradient Boosting</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
</tr>
<tr>
<td width="139">
<p>Ridge Regression</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
</tr>
<tr>
<td width="139">
<p>K-Nearest Neighbors</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
</tr>
<tr>
<td width="139">
<p>Passive Aggressive</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.37</p>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>

<table width="0"  align="">
<tbody>
<tr>
<td width="139">
<p><strong>Split 1 (70/30)</strong></p>
</td>
<td width="139">
<p><strong>Accuracy</strong></p>
</td>
<td width="139">
<p><strong>Precision</strong></p>
</td>
<td width="139">
<p><strong>Recall</strong></p>
</td>
<td width="139">
<p><strong>F1-Score</strong></p>
</td>
</tr>
<tr>
<td width="139">
<p>Decision Tree</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
</tr>
<tr>
<td width="139">
<p>Perceptron</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.46</p>
</td>
</tr>
<tr>
<td width="139">
<p>Naive Bayes</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
</tr>
<tr>
<td width="139">
<p>Logistic Regression</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM - Linear Kernel</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM - RBF Kernel</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.46</p>
</td>
</tr>
<tr>
<td width="139">
<p>Multilayer Perceptron</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.51</p>
</td>
</tr>
<tr>
<td width="139">
<p>Gradient Boosting</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
</tr>
<tr>
<td width="139">
<p>Ridge Regression</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
<td width="139">
<p>0.6</p>
</td>
</tr>
<tr>
<td width="139">
<p>K-Nearest Neighbors</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
<td width="139">
<p>0.51</p>
</td>
</tr>
<tr>
<td width="139">
<p>Passive Aggressive</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.52</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.45</p>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>

<table width="0"  align="">
<tbody>
<tr>
<td width="139">
<p><strong>Sprint 2 (80/20)</strong></p>
</td>
<td width="139">
<p><strong>Accuracy</strong></p>
</td>
<td width="139">
<p><strong>Precision</strong></p>
</td>
<td width="139">
<p><strong>Recall</strong></p>
</td>
<td width="139">
<p><strong>F1-Score</strong></p>
</td>
</tr>
<tr>
<td width="139">
<p>Decision Tree</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
<td width="139">
<p>0.62</p>
</td>
</tr>
<tr>
<td width="139">
<p>Perceptron</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.48</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.45</p>
</td>
</tr>
<tr>
<td width="139">
<p>Naive Bayes</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
<td width="139">
<p>0.55</p>
</td>
</tr>
<tr>
<td width="139">
<p>Logistic Regression</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM - Linear Kernel</p>
</td>
<td width="139">
<p>0.58</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.58</p>
</td>
<td width="139">
<p>0.58</p>
</td>
</tr>
<tr>
<td width="139">
<p>SVM with RBF Kernel</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.49</p>
</td>
<td width="139">
<p>0.46</p>
</td>
</tr>
<tr>
<td width="139">
<p>Multilayer Perceptron</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.53</p>
</td>
</tr>
<tr>
<td width="139">
<p>Gradient Boosting</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
<td width="139">
<p>0.61</p>
</td>
</tr>
<tr>
<td width="139">
<p>Ridge Regression</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
<td width="139">
<p>0.59</p>
</td>
</tr>
<tr>
<td width="139">
<p>K-Nearest Neighbors</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.5</p>
</td>
<td width="139">
<p>0.5</p>
</td>
</tr>
<tr>
<td width="139">
<p>Passive Aggressive</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.44</p>
</td>
<td width="139">
<p>0.53</p>
</td>
<td width="139">
<p>0.37</p>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>  

## Learning Curves for all models per split
<div align="">
	<img src="https://user-images.githubusercontent.com/60898339/234112473-6d06d936-2965-4813-ad0d-1cd5eda2b936.png">  
</div>
<div align="">
	<img src="https://user-images.githubusercontent.com/60898339/234112545-0ea1dfa6-3749-4de9-a523-34af293866c7.png">  
</div>
<div align="">
	<img src="https://user-images.githubusercontent.com/60898339/234112728-997cd7e5-7f47-4add-9fbb-dfdde8a18f7b.png">  
</div>
<div align="">
	<img src="https://user-images.githubusercontent.com/60898339/234113279-c8d297df-5a4a-44e9-bd9b-f1979262b2d5.png">  
</div>

## Best Performers
The best performing algorithms on this classification problem were Decision Tree (F1 of 0.62 on split 2), Gradient Boosting (F1 of 0.62 on Split 1) and Logistic regression (F1 of 0.60 on split 1). Overall, Decision Tree was chosen as the winning algorithm. 

Some of the poorer performing models include Perceptron and SVM with RBF Kernel, which consistently achieve scores worse than simply predicting a single class. The Passive aggressive classifier also tends to perform poorly, achieving an F1-Score below 0.4 in two of the three splits. 

## Citations

F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. 
### Related Work & Other Resources
* B. Resnick, "Researchers just released profile data on 70,000 OkCupid users without permission" Vox, May 12, 2016. [Online]. Available: https://www.vox.com/2016/5/12/11666116/70000-okcupid-users-data-release.
* E. Kirkegaard, "Intelligence and religiosity among dating site users" ResearchGate, December 2019. [Online]. Available: https://www.researchgate.net/publication/338125762_Intelligence_and_Religiosity_among_Dating_Site_Users.
* G. Suarez-Tangil, M. Edwards, C. Peersman, G. Stringhini, A. Rashid, M. Whitty, “Automatically Dismantling Online Dating Fraud”, 2020 IEEE Transactions on Information and Security
* C. van der Lee, T. van der Zanden, E. Krahmer, M. Mos, and A. Schouten, “Automatic identification of writers’ intentions: Comparing different methods for predicting relationship goals in online dating profile texts” in Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019), Nov. 2019, pp. 94–100. doi: 10.18653/v1/D19-5512.
* D. Boller, M. Lechner, and G. Okasa, The Effect of Sport in Online Dating: Evidence from Causal Machine Learning. 2021.
* I. Backus, "Predicting Gender from OKCupid profiles using ensemble methods" self published, Mar. 22, 2018. [Online]. Available: https://raw.githubusercontent.com/wiki/ibackus/okcupid-gender-prediction/okcupid_gender_prediction.pdf.
* M. Campbell, "Investigating OkCupid profiles: Can we predict someone's city?" Medium, July 21, 2022. [Online]. Available: https://medium.com/@macrotentional/investigating-okcupid-profiles-can-we-predict-someones-city-31a4734e96dd.

## Special Thanks
The 'star sign' team would like to thank our Professor **Dr. Juan M. Banda** for guiding us this semester! 
<a href="https://github.com/jmbanda"><img src="https://user-images.githubusercontent.com/60898339/222575865-617bc990-796a-4e29-834e-b30762f11526.png" width=25 height=25></a> 		
<a href="https://www.linkedin.com/in/jmbanda/"><img src="https://user-images.githubusercontent.com/60898339/222576175-1d3213f8-a001-4e7e-bb75-046fe5951fe3.png" width=25 height=25></a>


## Contributors  
<div align="">
	<tr>
		<td>
     <b>Mike Doan:</b>
		 <a href="https://github.com/manhtiendoan"><img src="https://user-images.githubusercontent.com/60898339/222575865-617bc990-796a-4e29-834e-b30762f11526.png" width=25 height=25></a>
		<a href="https://www.linkedin.com/in/manh-tien-doan/"><img src="https://user-images.githubusercontent.com/60898339/222576175-1d3213f8-a001-4e7e-bb75-046fe5951fe3.png" width=25 height=25></a> 
		</td>  
		<td>
    <b>&nbsp Jack Ericson:</b> 
		<a href="https://github.com/jackericson98"><img src="https://user-images.githubusercontent.com/60898339/222575865-617bc990-796a-4e29-834e-b30762f11526.png" width=25 height=25></a>
		<a href="https://www.linkedin.com/in/jackericson98/"><img  src="https://user-images.githubusercontent.com/60898339/222576175-1d3213f8-a001-4e7e-bb75-046fe5951fe3.png" width=25 height=25></a> 
		</td>  
		<td>
    <b>&nbsp Robert Tognoni:</b>
		<a href="https://github.com/rtogn"><img src="https://user-images.githubusercontent.com/60898339/222575865-617bc990-796a-4e29-834e-b30762f11526.png" width=25 height=25></a>
		<a href="https://www.linkedin.com/in/robert-tognoni-9a4795b0"><img  src="https://user-images.githubusercontent.com/60898339/222576175-1d3213f8-a001-4e7e-bb75-046fe5951fe3.png" width=25 height=25></a> 
		</td>  
	</tr>
</div>  

>> Contributions to this Github repository do not necessarily reflect contributions to the project as a whole. Code and textual content were worked on collaboratively by all team members, and then uploaded here for final submission
