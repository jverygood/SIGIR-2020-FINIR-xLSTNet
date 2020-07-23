## 1.the performance of the model

We have achieved 55.00658762, 70.28985507, 77.00922266 in task1, task2, and task3, with a total score of 67.43522178

## 2.the environment of the model

The complete environment we use is in the finir.yml file and runs on the Windows platform. Of course, you can also filter part of the environment yourself.

## 3.the model description

Our model is a combination of feature interaction and LSTNet. Using only LSTNet to predict whether the ups and downs have a general effect. We first use LSTNet to predict the closing price, and then judge the rise and fall, the effect is better, and the average accuracy of 60.80368906 can be reached. After adding the feature interaction module, the evaluation accuracy of 67.43522178 can be achieved.

## 4.the test time it cost

The model takes about 28 seconds to test.