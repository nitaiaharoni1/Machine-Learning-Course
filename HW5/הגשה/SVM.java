package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instances;


public class SVM {
	public SMO m_smo;

	public SVM()  {
		this.m_smo = new SMO();
	}

	public void setKernel(Kernel kernel) {
		this.m_smo.setKernel(kernel);
	}

	public void setC(double c) {
		this.m_smo.setC(c);
	}

	public double getC() {
		return this.m_smo.getC();
	}

	public void buildClassifier(Instances instances) throws Exception {
		m_smo.buildClassifier(instances);
	}

	public int[] calcConfusion(Instances instances) throws Exception {
		double classifiedValue;
		double realValue;
		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			classifiedValue = this.m_smo.classifyInstance(instances.instance(i));
			realValue = instances.instance(i).classValue();
			if (realValue == 1 && classifiedValue == 1) {
				TP++;
			} else if (realValue == 0 && classifiedValue == 0) {
				TN++;
			}else if (realValue == 0 && classifiedValue == 1) {
				FP++;
			}else {
				FN++;
			}
		}
		int[] confusion = { TP, FP, TN, FN };
		return confusion;
	}
	
	public double[] calcTprFpr(int[] conf){
		double TP = conf[0];
		double FP = conf[1];
		double TN = conf[2];
		double FN = conf[3];

		double tpr = TP/(FN+TP);
		double fpr = FP/(FP+TN);
		double[] tprFpr = {tpr, fpr};
		return tprFpr;
	}




}
