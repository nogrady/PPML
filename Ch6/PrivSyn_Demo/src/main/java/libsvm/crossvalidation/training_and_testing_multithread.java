package libsvm.crossvalidation;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import libsvm.crossvalidation.Cross_Validation_Runnable;

/** 
* @author Mingchen Li
* Sep 18, 2018
*/

public class training_and_testing_multithread {
	
	public static void main(String args[]) throws IOException{
		//--------------------------------Change Input parameter here---------------------------------------------------//
		String [] dataset_list = {"australian"}; 					// diabetes  breast-cancer  australian
		String scenario = "SynTR_OrgTE";   						// SynTR_OrgTE, OrgTR_SynTE	
		int[] k_list = new int[]{25, 50, 75, 100};				//  Sample size of each cluster
		double[] eplison_list = new double[]{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};		// Privacy budget of each dataset
		//--------------------------------------------------------------------------------------------------------------//
		
		training_and_testing_multithread t=new training_and_testing_multithread();
		for(String s:dataset_list){ 							// PrivSyn testing 
			for(int k:k_list){
				for(double e:eplison_list){
					for(int exp = 0; exp < 10; exp++)
					t.training(s,k,e, exp,scenario);
				}				
			}			
		}
		
	    System.out.println("Finishe SVM testing!");
	}
		
	public void training(String set,int k, double eplison, int exp, String scenario) throws IOException {
		(new File("Cross_Validation_Results.txt")).delete();
		int poolsize = 800;		
		int folder_cv= 10;
		
		double[] param_c={Math.pow(2, -10),Math.pow(2, -8), Math.pow(2, -6), Math.pow(2, -4),
				Math.pow(2, -2),Math.pow(2, 0),Math.pow(2, 2),Math.pow(2, 4),Math.pow(2, 6),Math.pow(2, 8),Math.pow(2, 10)};
		double[] param_g={Math.pow(2, -10), Math.pow(2, -8), Math.pow(2, -6), Math.pow(2, -4),
				Math.pow(2, -2),Math.pow(2, 0),Math.pow(2, 2),Math.pow(2, 4),Math.pow(2, 6),Math.pow(2, 8),Math.pow(2, 10)};
		
		/*
		 * double[] param_c={Math.pow(2, -4), Math.pow(2, -2), Math.pow(2, 0),
		 * Math.pow(2, 2), Math.pow(2, 4)}; double[] param_g={Math.pow(2, -4),
		 * Math.pow(2, -2), Math.pow(2, 0), Math.pow(2, 2), Math.pow(2, 4)};
		 */
		 
		String method = "PrivSyn";	
		 
		String FilePath = "./ExpData/" + set +"/"+ scenario + "/" ;  //TODO
		String testset = ""; 
		String dataset = "";
		
		if(scenario.equals("OrgTR_SynTE")) {
			dataset = FilePath + "NotSeed" + set + "_csv_svm";  
			testset = FilePath + method + "/" + set + "_syn_"+ k + "_" + Double.toString(eplison) + "_" + exp ; 			
		}else {
			dataset = FilePath + method + "/" + set + "_syn_"+ k + "_" + Double.toString(eplison) + "_" + exp ; 
			testset = FilePath + "NotSeed" + set + "_csv_svm";
		}
		
		File file = new File("");
		String currentDirectory = file.getAbsolutePath();
		System.out.println("Current working directory : "+currentDirectory);
		
		ExecutorService executor = Executors.newFixedThreadPool(poolsize);
		for(double c:param_c)
			for(double g:param_g) {	
				Cross_Validation_Runnable cv_run=new Cross_Validation_Runnable(c+"", g+"", folder_cv, dataset, dataset+"_model");
				executor.execute(cv_run);
			}
		
		executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all cross-validation threads!");        
        
		//get the best model
		BufferedReader br = new BufferedReader(new FileReader("Cross_Validation_Results.txt"));
		double accuracy = 0, best_c=2, best_g=2;
		String line="";
		while ((line=br.readLine()) != null) {
			String[] data=line.split("\t");
			if(accuracy < Double.parseDouble(data[2])){
				accuracy = Double.parseDouble(data[2]);
				best_c = Double.parseDouble(data[0]);
				best_g = Double.parseDouble(data[1]);		
			}
		}
		br.close();
		
		System.out.println(best_c+","+best_g+","+accuracy);
		
		//train again to get the best model with the highest accuracy model
		String[] trainArgs = {"-g", best_g+"", "-c", best_c+"", 
				dataset, dataset+"_model"};
		
			svm_train.main(trainArgs, "nouse");
		//Then test 
		System.out.println("g:"+best_g+"c:"+best_c+"f1score:"+accuracy);
		
		String[] testArgs = {testset, dataset+"_model", testset+"_result"};
		PrintWriter pw = new PrintWriter(new FileOutputStream("Result.txt", true));
		pw.print(dataset +","+ testset + "," + k +","+ eplison + "," + exp + ":");                                    //Test testset
		pw.flush();
		svm_predict.main(testArgs);
		
		pw.close();
		
	}
	
	public static String CSVtoLibSVM(String args) throws IOException {
		// TODO Auto-generated method stub
				
		String input_filepath = args;
		String outputfile = input_filepath + "_svm";
		
		BufferedReader br = new BufferedReader(new FileReader(input_filepath));
		PrintWriter pw = new PrintWriter(outputfile);
		String line="";
		br.readLine();
				
		while((line = br.readLine())!=null){
			String[] tmp = line.split(",");
			pw.print(tmp[0]+" "); //label
			pw.flush();
			for(int i=1; i< tmp.length;i++){
				if(i== tmp.length-1) {
					pw.println(i+":"+tmp[i]);
				}else {
					pw.print(i+":"+tmp[i]+" ");
				}
			}
			pw.flush();									
		}
		pw.close();
	return outputfile;
	}
	

}

