package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given
	 * dataset.
	 * 
	 * @param instances
	 *            The original dataset.
	 * @return A scaled instances object.
	 * @throws Exception
	 */
	public Instances scaleData(Instances instances) throws Exception {
		Standardize standardize = new Standardize();
		standardize.setInputFormat(instances);
		Instances scaledInstances = Filter.useFilter(instances, standardize);
		return scaledInstances;
	}
	

}