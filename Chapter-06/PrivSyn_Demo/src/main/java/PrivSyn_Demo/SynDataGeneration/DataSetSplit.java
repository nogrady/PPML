package PrivSyn_Demo.SynDataGeneration;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

/** 
* @author Mingchen Li
* May 11, 2021
*/

/* Function: Split the original dataset by label and generate the seed dataset for each label */

public class DataSetSplit {

	public static void main(String[] args) throws IOException {
	//--------------------------------Change Input parameter here---------------------------------------------------//
		String name = "breast-cancer";				// diabetes, breast-cancer, australian	
		String scenario = "SynTR_OrgTE";		// SynTR_OrgTE, OrgTR_SynTE	
		double seedRatio = 0.8;					// The ratio to get the number of seed data from the original dataset
		String label1 = "+1";  
		String filepath = "./ExpData/"+ name +"/" + scenario +"/";		// Input directory
	//-------------------------------------------------------------------------------------------------------------//
		
		String dataset = name + "_csv";		
		BufferedReader br = new BufferedReader( new FileReader(filepath + dataset));
		PrintWriter pw0 = new PrintWriter(filepath + "0Seed"+ dataset);
		PrintWriter pw1 = new PrintWriter(filepath + "1Seed" + dataset); 
		PrintWriter pwOther = new PrintWriter(filepath + "NotSeed" + dataset);

		ArrayList<String> List0 = new ArrayList<String>(); 
		ArrayList<String> List1 = new ArrayList<String>(); 
		
		String line = "";
		while((line = br.readLine())!=null){
			 String[] data = line.split(",");
			 if(data[0].equals(label1)){
				 List1.add(line); 
			 }else{
				List0.add(line);
			 }			
		}
		
		System.out.println("Num of Label 0:" + List0.size() );
		System.out.println("Num of Label 1:" + List1.size() );
		
		int numSeed0 = (int) (List0.size()*seedRatio);
		int numSeed1 = (int) (List1.size()*seedRatio);
		
		System.out.println("Num of seed 0:" + numSeed0 );
		System.out.println("Num of seed 1:" + numSeed1 );
		
		Random rand = new Random();
		for(int i = 0;i < numSeed0;i++) {				//generating seed dataset for label 0
			int index = rand.nextInt(List0.size());
			pw0.println(List0.get(index));
			List0.remove(index);
		}
		
		for(int i = 0;i < numSeed1;i++) {				//generating seed dataset for label 1 
			int index = rand.nextInt(List1.size());
			pw1.println(List1.get(index));
			List1.remove(index);
		}		
		
		pw1.flush();
		pw0.flush();
		pw1.close();
		pw0.close();
		
		for(String s:List0) {
			pwOther.println(s);
		}
		for(String s:List1) {
			pwOther.println(s);
		}
		
		pwOther.flush();
		pwOther.close();
		
	System.out.println("Finished");
	
	CSVToLibsvm.main(filepath + "NotSeed" + dataset);
	}

}
