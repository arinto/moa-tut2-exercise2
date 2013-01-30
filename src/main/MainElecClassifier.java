package main;

import moa.classifiers.Classifier;
import moa.classifiers.ElectricityClassifier;
import moa.evaluation.LearningCurve;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;

public class MainElecClassifier {

	/**
	 * @param args
	 */

	public static void main(String[] args) {

		//prepare classifier
		Classifier elecClassifier = new ElectricityClassifier();
		
		//prepare input file for streaming evaluation
		String arffFilePath = "/home/arinto/Dropbox/Thesis/MOA/DataSet/electricity/elecNormNew.arff";
		ArffFileStream electricityArff = new ArffFileStream(arffFilePath, -1);
		electricityArff.prepareForUse();
		
		//prepare classification performance evaluator
		WindowClassificationPerformanceEvaluator windowClassEvaluator = 
				new WindowClassificationPerformanceEvaluator();
		windowClassEvaluator.widthOption.setValue(1000);
		windowClassEvaluator.prepareForUse();
		
		//set EvaluatePrequential's parameter
		int maxInstances = 1000000;
		int timeLimit = -1;
		int sampleFrequencyOption = 1000;
		
		//do the learning and checking using evaluate-prequential technique
		EvaluatePrequential ep = new EvaluatePrequential();
		ep.instanceLimitOption.setValue(maxInstances);
		ep.learnerOption.setCurrentObject(elecClassifier);
		ep.streamOption.setCurrentObject(electricityArff);
		ep.sampleFrequencyOption.setValue(sampleFrequencyOption);
		ep.timeLimitOption.setValue(timeLimit);
		ep.evaluatorOption.setCurrentObject(windowClassEvaluator);
		ep.prepareForUse();
		
		LearningCurve le = (LearningCurve) ep.doTask();
		System.out.println(le);
	}
}
