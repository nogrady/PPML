package PrivSyn_Demo.SynDataGeneration;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
/** 
* @author Mingchen Li
* May 11, 2021
*/
public class CSVToLibsvm {

	public static void main(String args) throws IOException {
		// TODO Auto-generated method stub
				
		String input_filepath = args;
		String outputfile = input_filepath + "_svm";
		
		BufferedReader br = new BufferedReader(new FileReader(input_filepath));
		PrintWriter pw = new PrintWriter(outputfile);
		String line="";
				
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
	}
	
}
