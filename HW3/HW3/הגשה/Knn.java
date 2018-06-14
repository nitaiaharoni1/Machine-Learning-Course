package HomeWork3;

import java.awt.List;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import javax.swing.text.html.HTMLDocument.Iterator;

import HomeWork3.Knn.DistanceCheck;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

class DistanceCalculator {
	private double p;
	private DistanceCheck distanceCheck;

	public void setP(double p) {
		this.p = p;
	}

	public void setDistanceCheck(DistanceCheck distanceCheck) {
		this.distanceCheck = distanceCheck;
	}

	/**
	 * We leave it up to you wheter you want the distance method to get all relevant
	 * parameters(lp, efficient, etc..) or have it has a class variables.
	 */
	public double distance(Instance one, Instance two, double k, TreeMap<Double, LinkedList<Instance>> tree) {
		double distance;
		double threshold;
		if (this.distanceCheck.name() == "Regular") {
			if (p == Double.POSITIVE_INFINITY) {
				distance = lInfinityDistance(one, two);
			} else {
				distance = lpDisatnce(one, two);
			}
		} else {
			threshold = findThreshold(k,tree);
			if (p == Double.POSITIVE_INFINITY) {
				distance = efficientLInfinityDistance(one, two, threshold);
			} else {
				distance = efficientLpDisatnce(one, two, threshold);
			}
		}

		return distance;
	}
	
	private double findThreshold(double k, TreeMap<Double, LinkedList<Instance>> tree) {
		double threshold = Double.POSITIVE_INFINITY;
		int counter = 0;
		outerloop1:
		for (Map.Entry<Double, LinkedList<Instance>> ent : tree.entrySet()) {
			for (Instance i : ent.getValue()) {
				counter++;
				if (counter == k) {
					threshold = Math.pow(ent.getKey(), p);
                    break outerloop1;
				}
			}
		}
		return threshold;
	}

	/**
	 * Returns the Lp distance between 2 instances.
	 * 
	 * @param one
	 * @param two
	 */
	private double lpDisatnce(Instance one, Instance two) {
		double sum = 0;
		for (int i = 0; i < one.numAttributes() - 1; i++) {
			sum += Math.pow(Math.abs(one.value(i) - two.value(i)), this.p);
		}
		return Math.pow(sum, 1.0 / p);
	}

	/**
	 * Returns the L infinity distance between 2 instances.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double lInfinityDistance(Instance one, Instance two) {
		double max = 0;
		double temp;
		for (int i = 0; i < one.numAttributes() - 1; i++) {
			temp = Math.abs(one.value(i) - two.value(i));
			if (temp > max) {
				max = temp;
			}
		}
		return max;
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double efficientLpDisatnce(Instance one, Instance two, double threshold) {
		double sum = 0;
		for (int i = 0; i < one.numAttributes() - 1; i++) {
			sum += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
			if (sum > threshold) {
				return -1;
			}
		}
		return Math.pow(sum, 1.0 / p);
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
		double max = 0;
		double temp;
		for (int i = 0; i < one.numAttributes() - 1; i++) {
			temp = Math.abs(one.value(i) - two.value(i));
			if (temp > max) {
				max = temp;
				if (max >threshold) {
					return -1;
				}
			}
		}
		return max;
	}
}

public class Knn implements Classifier {

	public enum DistanceCheck {
		Regular, Efficient
	}

	private DistanceCheck distanceCheck;
	private Instances m_trainingInstances;
	private int weighted = -1;
	private double p;
	private int k;
	private long time; 
	
	

	public long getTime() {
		return time;
	}

	public void setDistanceCheck(DistanceCheck distanceCheck) {
		this.distanceCheck = distanceCheck;
	}

	public void setP(double parr) {
		this.p = parr;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setWeighted(int weighted) {
		this.weighted = weighted;
	}

	@Override
	/**
	 * Build the knn classifier. In our case, simply stores the given instances for
	 * later use in the prediction.
	 * 
	 * @param instances
	 */
	public void buildClassifier(Instances instances) throws Exception {
		this.m_trainingInstances = instances;
	}

	/**
	 * Returns the knn prediction on the given instance.
	 * 
	 * @param instance
	 * @return The instance predicted value.
	 */
	public double regressionPrediction(Instance instance) {
		TreeMap<Double, LinkedList<Instance>> tree = findNearestNeighbors(instance);
		double value;
		if (weighted == 1) {
			value = getWeightedAverageValue(tree);
		} else {
			value = getAverageValue(tree);
		}
		return value;
	}

	/**
	 * Caclcualtes the average error on a give set of instances. The average error
	 * is the average absolute error between the target value and the predicted
	 * value across all insatnces.
	 * 
	 * @param insatnces
	 * @return
	 */
	public double calcAvgError(Instances insatnces) {
		double avgErr = 0;
		double value = 0;
		int numInstances = insatnces.numInstances();
		for (int i = 0; i < numInstances; i++) {
			value = regressionPrediction(insatnces.instance(i));
			avgErr += Math.abs(value - insatnces.instance(i).classValue());
		}
		return (avgErr / numInstances);
	}

	/**
	 * Calculates the cross validation error, the average error on all folds.
	 * 
	 * @param insances
	 *            Insances used for the cross validation
	 * @param num_of_folds
	 *            The number of folds to use.
	 * @return The cross validation error.
	 * @throws Exception
	 */
	public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
		Instances training;
		Instances validation;
		this.time = 0;
		long start;
		double sum = 0;
		StratifiedRemoveFolds folds = new StratifiedRemoveFolds();
		folds.setNumFolds(num_of_folds);
		for (int i = 0; i < num_of_folds; i++) {
			folds.setFold(i + 1);
			folds.setInputFormat(instances);
			folds.setInvertSelection(true);
			training = Filter.useFilter(instances, folds);
			folds.setInputFormat(instances);
			folds.setInvertSelection(false);
			validation = Filter.useFilter(instances, folds);
			this.m_trainingInstances = training;
			start = System.nanoTime();
			sum += calcAvgError(validation);
			this.time += System.nanoTime() - start;
		}
		return sum / num_of_folds;
	}

	/**
	 * Finds the k nearest neighbors.
	 * 
	 * @param instance
	 * @return
	 */
	public TreeMap<Double, LinkedList<Instance>> findNearestNeighbors(Instance instance) {
		DistanceCalculator dist = new DistanceCalculator();
		dist.setP(this.p);
		dist.setDistanceCheck(this.distanceCheck);
		double key;
		Instance value;
		TreeMap<Double, LinkedList<Instance>> tree = new TreeMap<Double, LinkedList<Instance>>();
		TreeMap<Double, LinkedList<Instance>> retTree = new TreeMap<Double, LinkedList<Instance>>();
		LinkedList<Instance> instancesList;
		for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
			if (instance != m_trainingInstances.instance(i)) {
				key = dist.distance(instance, m_trainingInstances.instance(i), this.k, tree);
				if(key==-1) continue;
				if (!tree.containsKey(key)) {
					instancesList = new LinkedList<Instance>();
					instancesList.addFirst(m_trainingInstances.instance(i));
					tree.put(key, instancesList);
				} else {
					tree.get(key).addFirst(m_trainingInstances.instance(i));
				}
			}
		}
		int counter = 0;
		outerloop2:
		for (Map.Entry<Double, LinkedList<Instance>> ent : tree.entrySet()) {
			key = ent.getKey();
			while (tree.get(key).size() != 0) {
				value = tree.get(key).pollFirst();
				if (!retTree.containsKey(key)) {
					instancesList = new LinkedList<Instance>();
					instancesList.add(value);
					retTree.put(key, instancesList);
				} else {
					retTree.get(key).addFirst(value);
				}
				counter++;
				if (counter == k) {
					break outerloop2;
				}
			}
		}
		return retTree;
	}

	/**
	 * Cacluates the average value of the given elements in the collection.
	 * 
	 * @param
	 * @return
	 */
	public double getAverageValue(TreeMap<Double, LinkedList<Instance>> tree) {
		Instance listInstance;
		double sum = 0;
		int size = 0;
		for (Map.Entry<Double, LinkedList<Instance>> ent : tree.entrySet()) {
			while (ent.getValue().size() != 0) {
				listInstance = ent.getValue().pollFirst();
				sum += listInstance.classValue();
				size++;
			}
		}
		return sum / size;
	}

	/**
	 * Calculates the weighted average of the target values of all the elements in
	 * the collection with respect to their distance from a specific instance.
	 * 
	 * @return
	 */
	public double getWeightedAverageValue(TreeMap<Double, LinkedList<Instance>> tree) {
		Instance listInstance;
		double sum = 0;
		double w_iSum = 0;
		double w_i = 0;
		for (Map.Entry<Double, LinkedList<Instance>> ent : tree.entrySet()) {
			if (ent.getKey() != 0) {
				while (ent.getValue().size() != 0) {
					listInstance = ent.getValue().pollFirst();
					w_i = 1 / Math.pow(ent.getKey(), 2);
					sum += w_i * listInstance.classValue();
					w_iSum += w_i;
				}
			} else {
				listInstance = ent.getValue().pollFirst();
				return listInstance.classValue();
			}
		}
		return sum / w_iSum;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public double classifyInstance(Instance instance) {
		// TODO Auto-generated method stub - You can ignore.
		return 0.0;
	}

}
