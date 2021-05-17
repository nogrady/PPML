package PrivSyn_Demo.SynDataGeneration;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

/** 
* @author Mingchen Li
* May 11, 2021
*/
public class LibSVMtoCSV {
	//--------------------------------Change Input parameter here---------------------------------------------------//
	static String dataSetName = "australian";											// diabetes, breast-cancer, australian
	static int numofEXP = 10;															// Number of dataset generated from (k, epsilon) group
static String[] scenarioList = new String[]{"SynTR_OrgTE"};								// SynTR_OrgTE, OrgTR_SynTE	
	static String[] methodList = new String[] {"PrivSyn"};								// Proposed method
	static int[] k_list = new int[]{25, 50, 75, 100};									// Sample size of each cluster
	static double[] eplison_list = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};	// Privacy budget of each dataset
	//--------------------------------------------------------------------------------------------------------------//
	public static void main(String[] args) throws IOException{
		
		for(String scenario:scenarioList){
			for(String method:methodList){
				for(int clustersize:k_list){
					for(double eplison:eplison_list){
						for(int exp = 0; exp < numofEXP; exp++){
							
							String path = "ExpData/"+ dataSetName + "/" + scenario +"/"+ method +"/"; 
							String inputname = path+ dataSetName + "_syn_" + String.valueOf(clustersize) +"_"+ Double.toString(eplison) + "_" + exp;	
							String outputname = path+ dataSetName + "_syn_" + String.valueOf(clustersize) +"_"+ Double.toString(eplison) + "_" + exp+"_csv";	
							BufferedReader br = new BufferedReader(new FileReader(inputname));
							String linereader="";
							PrintWriter pw = new PrintWriter(outputname);
							
							while((linereader = br.readLine())!=null){
								String[] part1 = linereader.split(" ");
								String[] part2 = new String[2];
								String label = "";
								HashMap <String,String> output = new HashMap <>();
								label = part1[0];
								for(int k=1;k<part1.length;k++){
									part2 = part1[k].split(":");

									output.put(part2[0], part2[1]);
								}
								
								int column_number = part1.length-1; 
							    for(int i=1;i<=column_number;i++){
							    	if(!output.containsKey(String.valueOf(i))){	    	
							    	 output.put(String.valueOf(i), String.valueOf(0));
							    	}
							     
							    }

							    pw.print(label);
							    for(int j=1;j<=column_number;j++){
							    	pw.print(","+output.get(String.valueOf(j))); 
							    }
							    pw.println();
							    pw.flush();
							}
							
							br.close();
							pw.close();
						}
					}
				}
			}
		}

		System.out.println("All files are converted to CSV.");
		} 
}
