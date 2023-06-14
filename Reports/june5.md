# Summary

| Date   | Notes
| :----- | :-------------------------------
| 6/5  | Group meeting, started reading research papers, accessed datasets, larger group meeting
| 6/6  | Twinmotion vs. UE meeting, finished papers, started on building models using datasets
| 6/7  | Fixed issues with training model, started reading about pair programming, researched confusion matrix and accuracy. 
| 6/8  | Finished reading about pair programming, started reading research WriteUp, started researching about infernece.
| 6/9  | Updated weekly report, followed fastai tutorial to classify painting vs. drawing


# Activities
* Several group meetings where we discussed overarching goals and plans, twinmotion vs. unreal, and the weekly tasks. 

* Trained models on old datasets, had some issues with the results being only rotations and not commands, fixed them with the function from git. 

* Built a confusion matrix using old dataset

    * <img width="500" alt="handmadematrix3" src="https://github.com/daisy-abbott/ARCSLab-reports/assets/112681549/b351554e-0da7-410b-92f7-09a9dc60708f">

* Made an acuracy report using old dataset

    * <img width="500" alt="handmadematrix4" src="https://github.com/daisy-abbott/ARCSLab-reports/assets/112681549/06130abd-802e-4a63-806d-528cff26525e">
* Started to add suggestions to Research Write UP

* Read research papers and added questions to the doc

* Read pair programming articles

* Tried to make a model that classified art as painting or drawing which did not get great results 
    * <img width="500" alt="paintingsvdrawings" src="https://github.com/daisy-abbott/ARCSLab-reports/assets/112681549/c47e38aa-ccc2-4f49-ac37-86d02b41c6d9">
    * confusion matrix of this 
    * <img width="500" alt="Screenshot 2023-06-09 at 12 19 09 PM" src="https://github.com/daisy-abbott/ARCSLab-reports/assets/112681549/c42e0384-ce35-4ef0-8364-52f8b36c8015">

# Issues
* I think the data analysis from confusion matrix and accuracy report contradict each other. In the confusion matrix, we see that the model is really good at predicted forward when it should be forward, but there were 174 instances that it predicted forward when it should have been left, 146 instances where it predicted forward when it should have been right. There were also a number of times when the model predicted left or right when it should have been forward. Luckily it seldem predicted left when it should be right and vice versa. However the data from the classification report seems to somewhat contradict this. The overall accuracy was 91% but the precision for forward was tied with that of right, and both were worse than left. Which seems odd but I may also just not be understanding conceptually. 

* Ran into an issue with the painting vs. drawing and was recieving a cannot diplay batch because it is empty error and Anjali's code was super helpful in solving. And then also the acuracy was just really bad. 

# Plans 
* Continue looking at new models and experimenting with them. 

# Articlese

* Investigating Neural Network Techniques 
lots of questions on this one, very intersting to use prior command in helping decision making of next one.

* Pair Programming, helpful documentation for how ot use live share in VScode as well as solidifying the driver/navigator partnership
