# Hmode-Classifier-For-FDP
This is a classification tool designed to utilize ECE and PR diganostic data. There exists methods for using both individuall or as a combined Ensemble model.

To use the models, please see the jupyter notebook detailing how the data should look like and how best to use the functions presented. Getting the data to be formatted correctly is, I assume, the most challenging part of using this codebase; pleaes see how the data dictionaries are setup as described int he example code jupyter notebook.
The labels generated are provided as an excel sheet, however, the DIII-D must be obtained by the user. Please reach out to me if help is needed with managing data formatting or accessing data.

The ECE Model:  
![ECE RBF Flow Chart.pdf](https://github.com/user-attachments/files/28619657/ECE.RBF.Flow.Chart.pdf)


The PR Model:  
![PR_Flow_Chart.pdf](https://github.com/user-attachments/files/28619678/PR_Flow_Chart.pdf)


The Ensemble Model:  
![Ensemble Model image.pdf](https://github.com/user-attachments/files/28619672/Ensemble.Model.image.pdf)


Result of Ensemble Model on Randomized Shot Data:  
<img width="422" height="222" alt="Screenshot 2026-06-04 at 6 19 57 PM" src="https://github.com/user-attachments/assets/5eedeca8-6f3d-4b4b-8441-1952994ee871" />


References:  
Clark, Randall, et al. "Plasma confinement state classification in fusion power plants: Profile reflectometer and ensemble diagnostics." Nuclear Fusion (2026).  
Clark, Randall, et al. "Plasma confinement state classification via FPP relevant microwave diagnostics." Plasma Physics and Controlled Fusion 68.1 (2026): 015022.
