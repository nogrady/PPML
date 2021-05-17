package PrivSyn_Demo.SynDataGeneration;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import smile.clustering.HierarchicalClustering;
import smile.clustering.linkage.Linkage;
import smile.clustering.linkage.UPGMALinkage;
import smile.clustering.linkage.UPGMCLinkage;
import smile.clustering.linkage.WardLinkage;
import smile.math.*;
import smile.math.Math;
import PrivSyn_Demo.SynDataGeneration.generativeModel;

/** 
* @author Mingchen Li
* May 11, 2021
*/

/* Function: Get the indepedent feature set using hierarchical clusting */
	
public class getIFS_HierarchicalClustering {
	public static HashMap<Integer,ArrayList<Integer>> main(String args) throws IOException{
		String DataSetname = "./OriginalDataset/" + args + "_csv"; 		
		ArrayList<Record> orgrecords = generativeModel.readfile(DataSetname);	
		int NumOfFeature = orgrecords.get(0).length() - 1; 
		double numberOfData = (double)orgrecords.size(); 
		double[] mean =	generativeModel.getmean(orgrecords) ;
		double[] var = generativeModel.getVar(orgrecords);
		double[][] covarM = generativeModel.getcovarM(orgrecords);			
		
		double[][] proximity = new double[NumOfFeature][];		//distance matrix lower triangular matrix 		
		for (int i = 0; i < NumOfFeature; i++) {
			proximity[i] = new double[i+1];
		    for (int j = 0; j < i; j++){
		    	proximity[i][j] = 2*numberOfData - 2*numberOfData*(covarM[i][j]/(Math.sqrt(var[i])*Math.sqrt(var[j])));
		    }
		}
		
//		//System.out.println("proximity M:");
//		for(int i =0; i < proximity.length; i++){
//			System.out.println(Arrays.toString(proximity[i]));
//		}
		
		HierarchicalClustering hac = new HierarchicalClustering(new UPGMALinkage(proximity));
			
		int[][] tree = hac.getTree();
									
		for(int i = 0; i < tree.length; i++){
			for(int j = 0; j < tree[i].length ; j++){
				//System.out.println(tree[i][j]);
			}
			//System.out.println();
		}
					
		double[][] disM = covarM ;
		   
		for(int i = 0; i < covarM.length; i++ ){
			for(int j = 0; j < covarM.length; j++){
				disM[i][j] = 1 - covarM[i][j];
			}
		}
		
		double[][] distanceM = new double[NumOfFeature][NumOfFeature];
		for(int i = 0; i < NumOfFeature; i++){
			for(int j = 0; j < i; j++){
				distanceM[i][j] = proximity[i][j];
				distanceM[j][i] = distanceM[i][j];  
			}
		}
		 
		
		double minScore = Double.MAX_VALUE;
		int[] membership_minDB = new int[NumOfFeature]; 
		int NumCluster_minDB = 0;
		
		for(int i = 2; i < NumOfFeature; i++){
			int[] membership = hac.partition(i);
			double score = DaviesBouldinIndex(distanceM, membership, i);

			if(score < minScore){
				minScore = score;
				membership_minDB = membership;
				NumCluster_minDB = i;
			}
			
		}

		HashMap<Integer, ArrayList<Integer>> memberMap = new HashMap<Integer, ArrayList<Integer>>();
		
		for (int i =1; i < membership_minDB.length; i++) {
		    if(!memberMap.containsKey(membership_minDB[i])){
		    	memberMap.put(membership_minDB[i], new ArrayList<Integer>(Arrays.asList(i)));
		    }
		    else{
		    	ArrayList<Integer> tmp = memberMap.get(membership_minDB[i]);
		    	tmp.add(i);
		    	memberMap.put(membership_minDB[i], tmp);	
		    }
		}
		return memberMap;
	}
		
	public static void addcolumn () throws IOException{
		BufferedReader br = new BufferedReader( new FileReader("australian_scale_tmp"));
		PrintWriter pw = new PrintWriter("australian_add_scale_tmp");
		String line = "";
		while((line = br.readLine())!=null){
			String[] data = line.split(",");
			pw.println(line + "," + data[1]);
			pw.flush();
		}
		pw.close();
	}
	
	
	public static double DaviesBouldinIndex(double[][] data, int[] assignments, int number_of_clusters){

		int n=data.length; // number of data
		int[] clusterSize = new int[number_of_clusters];
				
		double[] S = new double[number_of_clusters];
		double[] ig = new double[number_of_clusters];
		double[][] M = new double[number_of_clusters][number_of_clusters];
		double[][] R = new double[number_of_clusters][number_of_clusters];
		double[][] bg =	new double[number_of_clusters][number_of_clusters];
		
		HashMap<Integer, Integer> memberMap = new HashMap<Integer, Integer>();
		
		for(int i = 0; i < assignments.length; i++){
			memberMap.put(i, assignments[i]);
			clusterSize[assignments[i]]++;
		}
		
		for(int i = 0; i < clusterSize.length;i++){
			if(clusterSize[i]==1){
				ig[i] = 1;
			}
		}
		
				
		for(int i = 0; i < assignments.length; i++){
			for(int j = i+1; j < assignments.length; j++){
				if(memberMap.get(i) == memberMap.get(j)){
					S[memberMap.get(i)] += data[i][j];
					ig[memberMap.get(i)]++;
				}else if(memberMap.get(i) > memberMap.get(j)){					
					M[memberMap.get(i)][memberMap.get(j)] += data[i][j];
					bg[memberMap.get(i)][memberMap.get(j)]++;			
				}else{
					M[memberMap.get(j)][memberMap.get(i)] += data[i][j];
					bg[memberMap.get(j)][memberMap.get(i)]++;
				}
				
			}
		}
					
		double[] D=new double[number_of_clusters];
		
		for(int i=0;i<number_of_clusters ;i++){
			for(int j=0;j<i;j++){
				R[i][j]=(S[i]/ig[i]+S[j]/ig[j])/(M[i][j]/bg[i][j]);
				R[j][i]=R[i][j];
			}
		}
		
		double DB=0;
		
		for(int i=0;i<number_of_clusters ;i++){
			for(int j=0;j<number_of_clusters;j++){
				if(D[i]<R[i][j] && i!=j){
					D[i]=R[i][j];
				}
			}
		DB=DB+D[i];
		}

		return DB/number_of_clusters;
	} 
	
	
	public static void normlize(String DataSetname) throws IOException{
		ArrayList<Record> orgrecords = generativeModel.readfile(DataSetname);			
		
		BufferedReader br = new BufferedReader(new FileReader(DataSetname));
		PrintWriter pw = new PrintWriter(new FileOutputStream(DataSetname+"_nomarlized"));
		String line ="";
		
		double[] mean =	generativeModel.getmean(orgrecords) ;
		double[] var = generativeModel.getVar(orgrecords) ;	
				
		while((line = br.readLine())!=null){
			
			String [] data = line.split(",");
			double [] n_data  = new double[data.length];
			n_data[0] =Double.parseDouble(data[0]); 
			
			for(int i=1; i < data.length; i++){
				double value = Double.parseDouble(data[i]);
				n_data[i] = (value-mean[i])/Math.sqrt(var[i]); 
			}
			
			for(int i=0 ;i < n_data.length; i++){
				if(i!=n_data.length-1){
					pw.print(n_data[i] + ",");
				}else{
					pw.println(n_data[i]);
				}
			}
			pw.flush();
		}
		
		pw.close();
	}
	
}

