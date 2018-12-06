# Data Scientist Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, you will first develop code for an image classifier built with PyTorch, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.


## Training Process

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
