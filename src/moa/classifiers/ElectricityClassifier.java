package moa.classifiers;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

/**
 * Electricity-based Classifier to determine whether the price is up or down.
 * Simple classifier which determine the resulting class based on previous class result.
 *
 * @author Arinto Murdopo
 *
 */

public class ElectricityClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	//protected variables to maintain the state of the classifier
	protected double previousClassValue;
		
	@Override
	public void resetLearningImpl() {
		previousClassValue = 0.0;
	}
	
	//main function, called for every training example
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		previousClassValue = inst.classValue();
	}
	
	@Override
	public double[] getVotesForInstance(Instance inst) {
		if(previousClassValue == 0.0)
		{
			return new double[]{1.0, 0.0};
		}
		else if (previousClassValue == 1.0)
		{
			return new double[]{0.0, 1.0};
		}
		return new double[]{0.0, 0.0};
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// do nothing here
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}
}
