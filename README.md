# Hmode-Classifier-For-FDP
This is a classification tool designed to utilize ECE and PR diganostic data. There exists methods for using both individuall or as a combined Ensemble model.

To use the models, please see the jupyter notebook detailing ECE, PR, and Ensemble usage. The labeller is structured to prioritize ease of use, requiring minimal user overseight. To guage the length of a session, running the ensemble model (which runs both ECE and PR) for 300 full shots at 100 ms intervals lasted about 1 hour.

The ECE Model:  
![ECE RBF Flow Chart.pdf](https://github.com/user-attachments/files/28619657/ECE.RBF.Flow.Chart.pdf)


The PR Model:  
![PR_Flow_Chart.pdf](https://github.com/user-attachments/files/28619678/PR_Flow_Chart.pdf)


The Ensemble Model:  
![Ensemble Model image.pdf](https://github.com/user-attachments/files/28619672/Ensemble.Model.image.pdf)


Result of Ensemble Model on Randomized Shot Data:  
<img width="422" height="222" alt="Screenshot 2026-06-04 at 6 19 57 PM" src="https://github.com/user-attachments/assets/5eedeca8-6f3d-4b4b-8441-1952994ee871" />

Version Control
!Successful Regression Tests conducted at the following Versions:
!NumPy - 1.26.4
!Scikit-Learn - 1.9.0
!ONNXRuntime - 1.26.0
!toksearch - 2.8.0
!toksearch_d3d - 0.9.8

Version Guidance
NumPy - 
Scikit-Learn - 
ONNXRuntime - must support ONNX opset 17
toksearch/toksearch_d3d - so long as the data is still able to be pulled and the formatting is consistent, version shouldn't matter (Should version control matter, please alert the dev to update this toolkit to be compatible with the most up to date version of toksearch/FDP)

References:  
Clark, Randall, et al. "Plasma confinement state classification in fusion power plants: Profile reflectometer and ensemble diagnostics." Nuclear Fusion (2026).  
Clark, Randall, et al. "Plasma confinement state classification via FPP relevant microwave diagnostics." Plasma Physics and Controlled Fusion 68.1 (2026): 015022.
