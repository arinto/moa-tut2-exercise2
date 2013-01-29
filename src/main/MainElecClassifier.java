package main;

import weka.core.Instance;
import moa.classifier.ElectricityClassifier;
import moa.classifiers.Classifier;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.evaluation.LearningEvaluation;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluateModel;
import moa.tasks.LearnModel;

public class MainElecClassifier {

	/**
	 * @param args
	 */

	public static void main(String[] args) {

		Classifier elecClassifier = new ElectricityClassifier();
		
		//prepare input file for streaming evaluation
		String arffFilePath = "/home/arinto/Dropbox/Thesis/MOA/DataSet/electricity/elecNormNew.arff";
		ArffFileStream electricityArff = new ArffFileStream(arffFilePath, -1);
		electricityArff.prepareForUse();
		
		int maxInstances = 10000000;
		int numPasses = 1;
		
		LearnModel lm = new LearnModel(elecClassifier, electricityArff, maxInstances, numPasses);
		Object resultingModel = lm.doTask();
		
		//2nd stream for evaluation, still use the same stream 
		String evalArffFilePath = "/home/arinto/Dropbox/Thesis/MOA/DataSet/electricity/elecNormNew.arff";
		ArffFileStream evalElectricityArff = new ArffFileStream(evalArffFilePath, -1);
		evalElectricityArff.prepareForUse();
		
		ClassificationPerformanceEvaluator evaluator = 
				new BasicClassificationPerformanceEvaluator();
		
		EvaluateModel em = new EvaluateModel();
		em.modelOption.setCurrentObject(resultingModel);
		em.streamOption.setCurrentObject(evalElectricityArff);
		em.maxInstancesOption.setValue(maxInstances);
		em.evaluatorOption.setCurrentObject(evaluator);
		LearningEvaluation resultingEvaluation = (LearningEvaluation) em.doTask();
		
		System.out.println("Learning on the given ARFF File, Evaluation on " +
				"the given ARFF file");
		System.out.println("ElectricityClassifier class");
		System.out.println(resultingEvaluation);
		
		double measuredAccuracy = resultingEvaluation.getMeasurements()[1].getValue();
		System.out.println("Measured accuracy from classifier= " + measuredAccuracy);
		
		//Sample code to verify the accuracy result
		System.out.println("Now, verification of the classifier result!");
		int instanceCounter = 0;
		int correctCounter = 0;
		double predictedClass = 0.0;
		ArffFileStream verificationArff = new ArffFileStream(arffFilePath, -1);
		
		while (verificationArff.hasMoreInstances()) {
			Instance elecInst = verificationArff.nextInstance();
			double classValue = elecInst.classValue();
			if(classValue == predictedClass){
				correctCounter++;
			}
			predictedClass = classValue;
			instanceCounter++;
		}
		
		double calculatedAccuracy = 100*((double)(correctCounter)/(double)(instanceCounter));
		System.out.println("Calculated accuracy = " + calculatedAccuracy);

		//Verification statement
		if(calculatedAccuracy == measuredAccuracy)
		{
			System.out.println("Classifier result is the same to verification result");
		}
		else
		{
			System.out.println("Classifier result is NOT the same to verification result");
		}
		
		
	}
}
