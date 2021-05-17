package PrivSyn_Demo.SynDataGeneration;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.jblas.DoubleMatrix;
import org.jblas.Eigen;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import PrivSyn_Demo.SynDataGeneration.getIFS_HierarchicalClustering;

/** 
* @author Mingchen Li
* May 11, 2021
*/

/* Function: Generate the synthetic dataset from seed datasets */

public class generativeModel{
	//--------------------------------Change Input parameter here---------------------------------------------------//
	static String dataSetName = "breast-cancer";			// diabetes, breast-cancer, australian
	static String scenario = "SynTR_OrgTE";			// SynTR_OrgTE, OrgTR_SynTE	
	static String method = "PrivSyn";				// Proposed method
 	
	static String scdir = "./ExpData/"+ dataSetName + "/" + scenario + "/"; 	// Seed dataset input directory
	static String outputpath = "./ExpData/" + dataSetName + "/" + scenario +"/"+ method +"/";	//Synthetic dataset output directory 
			
	static String output_name =	dataSetName + "_syn_";			// Output name of the synthetic dataset
	static int numoflabel = 2;									// Total class of the dataset				
	static int[] k_list = new int[]{25, 50, 75, 100};  	        //  Sample size of each cluster
	static double[]epsilon_list = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};  // Privacy budget of each dataset
	static int numofEXP = 10;										//Number of dataset to be generated from (k, epsilon) group
	//-------------------------------------------------------------------------------------------------------------//
	
	public static void main(String[] args) throws IOException,SingularMatrixException,NonPositiveDefiniteMatrixException{
			for(int clustersize:k_list){
				for(double eplison:epsilon_list){
					for(int exp = 0; exp < numofEXP; exp++){
						File synfile = new File(outputpath + output_name + String.valueOf(clustersize) + "_" + Double.toString(eplison) + "_" + exp);
						synfile.delete();
						for(int label = 0; label < numoflabel; label++){
								
						String dataset = scdir + label+ "Seed" + dataSetName +"_csv";
											
						ArrayList<ArrayList<Integer>> attrsubsets = new ArrayList<ArrayList<Integer>>();
						HashMap<Integer, ArrayList<Integer>> memberMap = getIFS_HierarchicalClustering.main(dataSetName);
	
						for (int i:memberMap.keySet()){
							attrsubsets.add(memberMap.get(i));						
						}
																
						ArrayList<Record> orgrecords = readfile(dataset);
						int attrnum =  orgrecords.get(0).length() - 1; 
						//System.out.println("attr:"+attrnum);
						ArrayList<Record> subrecords = new ArrayList<Record>();
						ArrayList<Record> synrecords = new ArrayList<Record>();
						int numofdata = orgrecords.size();
						//System.out.println("numofdata:"+ numofdata);
						ArrayList<ArrayList<Record>> clusters = new ArrayList<ArrayList<Record>>();
						build(orgrecords);   											//set the index
						synrecords = orgrecords;
															
						for(int ii=0; ii<attrsubsets.size(); ii++){                    //for each attr subset
							for(int kk=0; kk <orgrecords.size(); kk++){					// for each record
								
								double[] subattrs = new double[attrsubsets.get(ii).size()]; 
								
								for(int jj=0; jj<subattrs.length; jj++){ 					
									double[] fulldata = orgrecords.get(kk).getData();	 			 //get full dimension data
									subattrs[jj] = fulldata[(int) attrsubsets.get(ii).get(jj)] ;      //extract the subset feature 				
								}
								Record tmp = new Record(subattrs);                                 
								subrecords.add(tmp);
							}
							
							build(subrecords);
							
							microaggregation MA = new microaggregation();
							clusters = MA.MDAV(subrecords,clustersize);
	
							subrecords.clear();
							
							
							double weight_eps = eplison/attrnum*(attrsubsets.get(ii).size()); 	//eps weight for each attrsubset should based on the subset size
													
							//System.out.println("Sampling subset"+ii+", clusters in this subset:"+ clusters.size());
							
							for(int i=0;i<clusters.size();i++){          //for each cluster
								
								//System.out.println("cluster"+i+",number of data in this cluster:"+clusters.get(i).size() );
								
								double[] mean = getmean(clusters.get(i));
								double[][] covarM = getcovarM(clusters.get(i));
								double[][] noise = new double[covarM.length][covarM.length];
								
								double d = attrsubsets.get(ii).size();
								//System.out.println("d:" +d);
								double C = clusters.get(i).size();
								//System.out.println("C:" +C);
								double lp_sd = 2*d/(numofdata* C* weight_eps* 0.4);   // laplace distribution b for mean 
								double lp_sd_covar =d*(d+1)/(numofdata*C*weight_eps*0.6);   // laplace distribution b for covarM
								//System.out.println("DP_model_parameter:" + lp_sd + "," + lp_sd_covar);
								
								LaplaceDistribution lp = new LaplaceDistribution(0, lp_sd);
								
								LaplaceDistribution lp_covar = new LaplaceDistribution(0, lp_sd_covar);
								
								PrintWriter pas = new PrintWriter(new FileOutputStream("check.txt"));
								
								for(int j=0;j<mean.length;j++){    					//add lplace noise to mean
									mean[j] = mean[j]+lp.sample();
								}
								
								for(int j = 0;j<covarM.length;j++){				
									pas.println("CovarM:" + Arrays.toString(covarM[j]));
								}	
								
								for(int j = 0;j<noise.length;j++){						//Generate Laplace Noise for CovarM
									for(int k = 0;k<=j;k++){					
										noise[j][k] = lp_covar.sample();
										noise[k][j] = noise[j][k];
									}
								}
								
								for(int j = 0;j<covarM.length;j++){						//add DP Noise to covarM
									for(int k = 0;k<covarM.length;k++){
											covarM[j][k]=covarM[j][k]+noise[j][k];																		
									}
								}																		

								double fixvalue =verify(covarM);

								//System.out.println("MinEigenValue:"+fixvalue);
								for(int j = 0;j<covarM.length;j++){
									covarM[j][j]=covarM[j][j] - 10*fixvalue;
								}
								
								
								for(int j = 0;j<noise.length;j++){
									pas.println("noise:" + Arrays.toString(noise[j]));
								}
								
								for(int j = 0;j<covarM.length;j++){
									pas.println("covar_DP:" + Arrays.toString(covarM[j]));
								}
								pas.flush();
								pas.close();
								
								;
								
								boolean repeat = true;	
								
								while(repeat == true){
									try{
										repeat = false;
										MultivariateNormalDistribution nmd = new MultivariateNormalDistribution(mean, covarM);
										}catch(NonPositiveDefiniteMatrixException | SingularMatrixException e){
											repeat = true;
											System.out.println("REGENERATING...REASON:" + e );
											covarM = getcovarM(clusters.get(i));
											
											for(int j = 0;j<noise.length;j++){						//Regenerate Laplace Noise for CovarM
												for(int k = 0;k<=j;k++){					
													noise[j][k] = lp_covar.sample();
													noise[k][j] = noise[j][k];
												}
											}
											
											for(int j = 0;j<covarM.length;j++){						//add the new generated DP Noise to covarM
												for(int k = 0;k<covarM.length;k++){					
													covarM[j][k]=covarM[j][k]+noise[j][k];
												}
											}
											
										}
								}
								
								MultivariateNormalDistribution nmd = new MultivariateNormalDistribution(mean, covarM);					
								
								for(int j=0;j<clusters.get(i).size();j++){              //for data in each cluster
									int pos =clusters.get(i).get(j).getPos();
									 
									double [] newdata = new double[synrecords.get(pos).getData().length];  
									newdata = synrecords.get(pos).getData();       //full dimension
	
									double[] tmp1 = nmd.sample(); //sub
	
									
									for(int k=0;k< tmp1.length;k++){
										newdata[(int) attrsubsets.get(ii).get(k)] = tmp1[k];
									}
								}
								
								
							}
						}
					
						PrintWriter pw = new PrintWriter(new FileOutputStream(outputpath + output_name + String.valueOf(clustersize) + "_" + Double.toString(eplison) + "_" + exp, true));	
	
						for(int s=0; s<synrecords.size();s++){  				//SVM testing format
							double[] tmp= synrecords.get(s).getData();
							for(int ss=0; ss<tmp.length;ss++){
								if(ss==0){
									pw.print(tmp[ss]+" ");
								}else if(ss==(tmp.length-1)){
									pw.print(ss+":"+tmp[ss]);
								}else{
									pw.print(ss+":"+tmp[ss]+" ");
								}
							}
							pw.println();
	
						}
						
						pw.flush();
						pw.close();					
					}
				  }	
				System.out.println(dataSetName + " synthetic data generation finished: k=" + String.valueOf(clustersize) + ", epsilon=" + Double.toString(eplison));	
				}
			}
}
	
//Below are methods	
	
	public static ArrayList<Record> build(ArrayList<Record> input) throws IOException{
		ArrayList<Record> samplebag = new ArrayList<Record>();
		for(int i=0;i<input.size();i++){
			input.get(i).setPos(i);			
		}
		return samplebag;
	}
	
	public static ArrayList<Record> readfile(String input_filepath) throws IOException{ //read label
		String line="";
		int pos =0;
		int count=0;
		ArrayList<Record> samplebag = new ArrayList<Record>();
		BufferedReader br = new BufferedReader(new FileReader(input_filepath));
		while((line = br.readLine())!=null){
			count++;
			String[] temp = line.split(",");
			double[] sample = new double[temp.length];
			
			for(int i = 0; i<sample.length;i++){
				sample[i] =  Double.parseDouble(temp[i]);				
			}
			
			Record r = new Record(sample,pos);
			pos++;
			samplebag.add(r);
		}
		//System.out.println("number of data:"+ count);
		return samplebag;
	} 
	
	public static double[] getmean(ArrayList<Record> input) throws IOException{
		double[] mean = new double[input.get(0).getData().length]; 
		double sum =0.0;
		for(int i=0;i<mean.length;i++){
			for(int j=0;j<input.size();j++){
				double[] tmp = input.get(j).getData();
				sum+=tmp[i]; 
			}
			mean[i]=sum/input.size();
			sum=0.0;
		}
		//System.out.println("meannnn vector:"+Arrays.toString(mean));
		return mean;		
	}
		  
	public static double[] getVar(ArrayList<Record> input) throws IOException{
		double[] mean = getmean(input);
		double[] var = new double[mean.length];
		double sum=0;
		for(int i= 0; i< var.length; i++){
			for(int k=0;k<input.size();k++){
				double[] tmp =input.get(k).getData();
				sum +=  Math.pow(tmp[i]-mean[i],2);					
			}
			//System.out.println(sum);
			var[i] =sum/input.size();
			sum=0;
		}
		return var;
	}
	
	public static double verify(double[][] covarM) throws IOException{
		DoubleMatrix DM =new DoubleMatrix();
	    DoubleMatrix TestCovar = DM.zeros(covarM.length, covarM.length);

		for(int i=0;i<covarM.length;i++){
	    	for(int j=0;j<covarM.length;j++){
	    		  TestCovar.put(i,j, covarM[i][j]);	    	  
	    	}
	     }
		
		Eigen EigenFunc =new Eigen();
		DoubleMatrix eigenValue = EigenFunc.symmetricEigenvalues(TestCovar);
		double mineigenvalue =0;

		for(int j=0;j<eigenValue.rows;j++){	    			
	    	if(eigenValue.get(j)<mineigenvalue)			
	    		mineigenvalue = eigenValue.get(j);	    		 
	    }
		return mineigenvalue;//System.out.println();
	}
		
	public static double[][] getcovarM(ArrayList<Record> input) throws IOException{
		double[][] covarM = new double[input.get(0).getData().length][input.get(0).getData().length];		
		double[] mean = getmean(input);
		double sum = 0.0;

		for(int i=0;i<input.get(0).getData().length;i++){
			for(int j=0;j<input.get(0).getData().length;j++){
				for(int k=0;k<input.size();k++){
					double[] temp = input.get(k).getData();
					sum += (temp[i]-mean[i])*(temp[j]-mean[j]);   
				}
				covarM[i][j] = sum/(input.size());
				//TODO
				/*if(covarM[i][j]==0){
					covarM[i][j] = Math.pow(10, -8);
				}*/
				sum=0.0;
			}
		}
	
		return covarM;		
	}
}
