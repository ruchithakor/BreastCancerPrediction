# Breast Cancer Prediction

Breast cancer is the most common type of cancer among women worldwide. With early diagnosis, one can start timely clinical treatment and improve the chances of survival significantly. Also. accurate classification of benign tumors can prevent patients undergoing unnecessary treatments. Because of its unique advantages in critical feature detection from complex datsets, Machine learning is widely used for pattern classification and forecast modelling.

Diagnosis of Breast cancer is performed when an abnormal lump is found in an x-ray or by self-examination. Mammography is an x-ray of the breast and is the most important screening test for the diagnosis. Women aged 40-45 years or older are usually at average risk of breast cancer and should have a mammogram once a year. Women with high risk should have yearly mammograms with MRI starting at age 30.

Most of the cases of breast cancer cannot be linked to a specific cause, but there are some of the known risk factors for the disease. 
The chance of getting breast cancer increases as women age. Women with breast cancer in one breast are at risk of developing cancer in the other breast too. A family history of breast cancer(if a woman's mother, sister or daughter has breast cancer) also raises the risk of getting it. Childbearing and menstrual history also affects the chances of getting breast cancer. 

### objective

With this analysis we can find features that are most helpful in predicting malignant or benign cancer and see general trends that may help us in model selection. The main goal is to classify whether the breast cancer is benign or malignant. Machine learning classification methods have been used to fit the function that can predict for the new input.

### Dataset

Dataset used here is publicly available and was created by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison, Wisconsin, USA. fluid samples were takedn from patients for creating the database. Graphical programme called Xcyt was used to perform analysis of features of samples based on digital scan. This programme used curve-fitting algorithm to compute ten features from each one of the cells in the sample and then it calculates the mean value, standard error and extreme value. 

Ten features computed for each cells:

1. radius : mean of distances from center to points on the perimeter
2. texture : standard deviation of gray-scale values
3. perimeter
4. area
5. smoothness : local variation in radius lengths
6. compactness : perimeter² / area — 1.0
7. concavity : severity of concave portions of the contour
8. concave points : number of concave portions of the contour
9. symmetry
10. fractal dimension : “coastline approximation” — 1
