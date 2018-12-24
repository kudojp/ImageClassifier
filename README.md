# Project Overview

This is a 1st project of Udacity's Data Scientist Nanodegree program. In this project, I first developed code in Jupyter Notebook for an image classifier built with PyTorch following the instruction of Udacity. I converted it into a command line application which enables us to predict the type of flower from the image file.


# Files
* train.py (file to create the deep learning model)
* predict.py (file to predict the type of flower from the image file)



# Training Process

Below is the history of training process with vgg13. I stopped training after 15 epochs and testing accuracy was around 88%.

$ python train.py flowers/ --save checkpoint.pth --epochs 10 --gpu

curently 1th epoch
  training   : loss = 2.4684901237487793, accuracy = 0.41783536585365855
  validation : loss = 1.035745620727539, accuracy = 0.6797542732495528

curently 2th epoch
  training   : loss = 1.1039330959320068, accuracy = 0.7030487804878048
  validation : loss = 0.5928100943565369, accuracy = 0.8249198725590339

curently 3th epoch
  training   : loss = 0.8355001211166382, accuracy = 0.7673780487804878
  validation : loss = 0.5692702531814575, accuracy = 0.8389423076923077

curently 4th epoch
  training   : loss = 0.7327343225479126, accuracy = 0.7981707317073171
  validation : loss = 0.4215972423553467, accuracy = 0.8740651699212881

curently 5th epoch
  training   : loss = 0.6132146716117859, accuracy = 0.8287601625047079
  validation : loss = 0.3322969377040863, accuracy = 0.9122596153846154

curently 6th epoch
  training   : loss = 0.5563485026359558, accuracy = 0.8423780487804878
  validation : loss = 0.53471440076828, accuracy = 0.8695245729042933

curently 7th epoch
  training   : loss = 0.5453370809555054, accuracy = 0.8484756097560976
  validation : loss = 0.452088326215744, accuracy = 0.8846153846153846

curently 8th epoch
  training   : loss = 0.5051045417785645, accuracy = 0.8560975609756097
  validation : loss = 0.31002840399742126, accuracy = 0.9173344006905189

curently 9th epoch
  training   : loss = 0.46340084075927734, accuracy = 0.8664126015290982
  validation : loss = 0.33820977807044983, accuracy = 0.9098557692307693

curently 10th epoch
  training   : loss = 0.4293689727783203, accuracy = 0.8778963414634147
  validation : loss = 0.33374178409576416, accuracy = 0.9091880344427549


training finished
  testing   : loss = 0.012962741777300835, accuracy = 0.886955976486206

checkpoint saved in checkpoint.pth
