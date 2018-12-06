# Data Scientist Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, you will first develop code for an image classifier built with PyTorch, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.


## Training Process

Below is the history of training process with vgg13. I stopped training after 15 epochs and testing accuracy was around 88%.

$ python train.py flowers/ --save checkpoint.pth --arch vgg13 --learning_rate 0.05 --hidden_units 1024 --epochs 15 --gpu

traing using cuda

curently 1th epoch
  training   : loss = 2.3467164039611816, accuracy = 0.4413109756097561
  validation : loss = 0.9845077395439148, accuracy = 0.7234241458085867

curently 2th epoch
  training   : loss = 1.0547101497650146, accuracy = 0.7192073170731708
  validation : loss = 0.5220341682434082, accuracy = 0.86017628128712

curently 3th epoch
  training   : loss = 0.7953851222991943, accuracy = 0.7783028454315372
  validation : loss = 0.45335015654563904, accuracy = 0.8721955120563507

curently 4th epoch
  training   : loss = 0.6430792808532715, accuracy = 0.8153963414634147
  validation : loss = 0.39407235383987427, accuracy = 0.89142628128712

curently 5th epoch
  training   : loss = 0.5708428621292114, accuracy = 0.8391260161632444
  validation : loss = 0.5864042043685913, accuracy = 0.8514957267504472

curently 6th epoch
  training   : loss = 0.4953936040401459, accuracy = 0.8584349594465116
  validation : loss = 0.38784322142601013, accuracy = 0.8986378197486584

curently 7th epoch
  training   : loss = 0.47287535667419434, accuracy = 0.8679369917730005
  validation : loss = 0.3692706227302551, accuracy = 0.9034455120563507

curently 8th epoch
  training   : loss = 0.46901360154151917, accuracy = 0.8675813009099262
  validation : loss = 0.3236832618713379, accuracy = 0.9113247853059036

curently 9th epoch
  training   : loss = 0.4564618766307831, accuracy = 0.870579268292683
  validation : loss = 0.43957045674324036, accuracy = 0.9013087611932021

curently 10th epoch
  training   : loss = 0.40806683897972107, accuracy = 0.8839430893339761
  validation : loss = 0.3157767951488495, accuracy = 0.9170673076923077

curently 11th epoch
  training   : loss = 0.3974379003047943, accuracy = 0.8889735771388543
  validation : loss = 0.3377111852169037, accuracy = 0.9149305545366727

curently 12th epoch
  training   : loss = 0.3619895279407501, accuracy = 0.8917682926829268
  validation : loss = 0.320803701877594, accuracy = 0.918536323767442

curently 13th epoch
  training   : loss = 0.3789733946323395, accuracy = 0.8919715448123653
  validation : loss = 0.3656074106693268, accuracy = 0.9170673076923077

curently 14th epoch
  training   : loss = 0.3358779549598694, accuracy = 0.9051321137242201
  validation : loss = 0.3245610296726227, accuracy = 0.9245459391520574

curently 15th epoch
  training   : loss = 0.3320297598838806, accuracy = 0.9028455283583664
  validation : loss = 0.3652203679084778, accuracy = 0.9162660264051877


training finished
  testing   : loss = 0.025311965495347977, accuracy = 0.8989752531051636

checkpoint saved in checkpoint.pth
