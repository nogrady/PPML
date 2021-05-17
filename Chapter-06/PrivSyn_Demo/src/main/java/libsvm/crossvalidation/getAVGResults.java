package libsvm.crossvalidation;

/** 
* @author Mingchen Li
* Sep 16, 2019
*/

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class getAVGResults {
	public static void main(String[] args) throws IOException {
		//--------------------------------Change Input parameter here---------------------------------------------------//
		String method = "PrivSyn";   						// Proposed method
		String datasetName = "australian";					// diabetes, breast-cancer, australian
		String scenario = "SynTR";							// SynTR, OrgTR	
		int numofEXP = 10;
		double[] eplison_list = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};		   // Privacy budget of each dataset
		String dataset = "./" + "Result_" + datasetName + "_" + scenario + ".txt";					   // Input file path
		PrintWriter pw1 = new PrintWriter("./" + datasetName + "_" + method + "_" + scenario + "_avgresult_k.csv"); // Output sort by k
		BufferedReader br = new BufferedReader( new FileReader(dataset)); 
		PrintWriter pw = new PrintWriter("./" + datasetName + "_" + method + "_" + scenario + "_avgresult_eps.csv"); // Output sort by eps
		//-------------------------------------------------------------------------------------------------------------//

		String line = "";
		
		int count = 0;
		double sum = 0.0;
		double eplison = 0.0;
		String k = "";
		
		while((line = br.readLine())!=null){ 
			count++;			
			if(count%(numofEXP*2) !=0){  //TODO divided by num of exp*2
				if(count%2 != 0){	
					String[] data = line.split(":");
					String[] param =  data[0].split(",");
					k = param[2];
					eplison = Double.parseDouble(param[3]);
					String[] data1 = data[1].split(" ");
					String[] data2 = data1[0].split("%");
					
					sum += Double.parseDouble(data2[0]);
					
				}
			}
			else{
			
				pw.println(k+"\t"+eplison+"\t"+sum/numofEXP);
				sum = 0;
				
			}
			
		}
		
		count = 0;

		for(double e:eplison_list){
			BufferedReader br1 = new BufferedReader( new FileReader(dataset));
			count = 0;
			sum = 0.0;
			while((line = br1.readLine())!=null){ 
				count++;			
				if(count%(numofEXP*2) !=0){
					if(count%2 != 0){	
						String[] data = line.split(":");
						String[] param =  data[0].split(",");
						k = param[2];
						eplison = Double.parseDouble(param[3]);
						String[] data1 = data[1].split(" ");
						String[] data2 = data1[0].split("%");							
						sum += Double.parseDouble(data2[0]);
				
					}
				}
				else{
					if(eplison == e){
						pw1.println(k+"\t"+eplison+"\t"+sum/numofEXP);							
					}
					sum = 0;
				}
				
			}
		br1.close();
		}

		pw.flush();
		pw1.flush();
		pw.close();
		pw1.close();
		System.out.println("Finished");
	}

}


