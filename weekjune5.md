# Summary

| Date   | Notes
| :----- | :-------------------------------
| 6/5  | Group meeting, started reading research papers, accessed datasets, larger group meeting
| 6/6  | Twinmotion vs. UE meeting, finished papers, started on building models using datasets
| 6/7  | Fixed issues with training model, started reading about pair programming, researched confusion matrix and accuracy. 
| 6/8  | Finished reading about pair programming, started reading research WriteUp, started researching about infernece.
| 6/9  | Updated weekly report, tried to implement inference models using old datasets


# Activities
* Several group meetings where we discussed overarching goals and plans, twinmotion vs. unreal, and the weekly tasks. 

* Trained models on old datasets, had some issues with the results being only rotations and not commands, fixed them with the function from git. 

* Built a confusion matrix using old dataset

    * <img width="500" alt="handmadematrix3" src="https://github.com/daisy-abbott/weekly-update/assets/112681549/24d6ac2e-f69c-4da5-bf39-1ab67116b758">

* Made an acuracy report using old dataset

    * <img width="500" alt="handmadematrix4" src="https://github.com/daisy-abbott/weekly-update/assets/112681549/9d394290-6189-468e-85dc-94e4f8470826">

* Started to add suggestions to Research Write UP

* Read research papers and added questions to the doc

# Issues
* I think the data analysis from confusion matrix and accuracy report contradict each other. 

In the confusion matrix, we see that the model is really good at predicted forward when it should be forward, but there were 174 instances that it predicted forward when it should have been left, 146 instances where it predicted forward when it should have been right. There were also a number of times when the model predicted left or right when it should have been forward. Luckily it seldem predicted left when it should be right and vice versa. 

However the data from the classification report seem to somewhat contradict these this. The overall accuracy was 91% but the precision for forward was tied with that of right, and both were worse than left. Which seems odd but I may also just not be understanding conceptually. 

# Plans 
* Continue looking at new models and experimenting with them. 

# Articlese

