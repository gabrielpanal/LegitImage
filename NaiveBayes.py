import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.ClassPrior = {}
        self.mean = {}
        self.variance = {}
        
        self.xTrain = np.array
        self.yTrain = np.array
          
    def _MeanVariance(self):
        temp = np.column_stack((self.xTrain, self.yTrain))
        epsilon = 1e-9
        epsilon = float(epsilon)
        for uniqueClass in np.unique(self.yTrain):
            for uniqueFeature in range(self.xTrain.shape[1]):
                self.mean[uniqueClass, uniqueFeature] = np.mean(temp[temp[:, -1] == uniqueClass, uniqueFeature])
                self.variance[uniqueClass, uniqueFeature] = np.var(temp[temp[:, -1] == uniqueClass, uniqueFeature], ddof=1) + epsilon

    def _PriorProb(self):
        for uniqueClass in np.unique(self.yTrain):
            self.ClassPrior[uniqueClass] = (self.yTrain == uniqueClass).sum()/self.yTrain.shape[0]

    def _GaussianDistribution(self, x, mean, variance):
        gd = (1 / np.sqrt(2 * np.pi * variance)) * np.exp((-(x-mean)**2) / (2 * variance))
        return gd
            
    def Train(self, X, y):
        self.xTrain = X
        self.yTrain = y

        self._MeanVariance()
        self._PriorProb()

    def Predict(self, X_test):
        postProbs = {}
        evidence = {}

        for sample in range(len(X_test)):
            evidence[sample] = 0.0
            log = 0
            for uniqueClass in np.unique(self.yTrain):
                probabilities = []
                for uniqueFeature in range(self.xTrain.shape[1]):
                    probability = self._GaussianDistribution(X_test[sample, uniqueFeature], self.mean[uniqueClass, uniqueFeature], self.variance[uniqueClass, uniqueFeature])
                    probabilities.append(probability)
                
                epsilon = 1e-10  # or another small positive value
                postProbDefault = np.prod(probabilities) * self.ClassPrior[uniqueClass]
                postProbLog = np.sum(np.log(np.array(probabilities) + epsilon)) + np.log(self.ClassPrior[uniqueClass])

                if not np.isfinite(postProbDefault) or postProbDefault == 0 or log == 1:
                    postProb = postProbLog
                    log = 1
                else:
                    postProb = postProbDefault

                postProbs[sample, uniqueClass] = postProb
                evidence[sample] += postProb

            # Handle the case where evidence is zero
            if evidence[sample] == 0.0:
                # Set a default value, 0.5 indicating uncertainty
                for uniqueClass in np.unique(self.yTrain):
                    postProbs[sample, uniqueClass] = 0.5
            else:
                # Normalize the probabilities if evidence is non-zero
                if log == 1:
                    for uniqueClass in np.unique(self.yTrain):
                        postProbs[sample, uniqueClass] = np.exp(postProbs[sample, uniqueClass]) / evidence[sample]
                else:  
                    for uniqueClass in np.unique(self.yTrain):
                        postProbs[sample, uniqueClass] = postProbs[sample, uniqueClass] / evidence[sample]
        
        authenticProb = postProbs.get((0, 0)) * 100
        manipulatedProb = postProbs.get((0, 1)) * 100

        outProb = "Authentic: {:.2f}% Manipulated: {:.2f}%".format(authenticProb, manipulatedProb)
        print(outProb)    
        
        predictedClasses = {}    
        for sample in range(len(X_test)):
            max_class = max(np.unique(self.yTrain), key=lambda uniqueClass: postProbs[sample, uniqueClass])
            predictedClasses[sample] = max_class

        predictions = np.array(list(predictedClasses.values()))

        return predictions, outProb