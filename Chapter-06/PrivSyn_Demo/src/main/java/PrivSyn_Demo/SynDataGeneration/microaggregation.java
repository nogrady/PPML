package PrivSyn_Demo.SynDataGeneration;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import PrivSyn_Demo.SynDataGeneration.Record;
/** 
* @author Mingchen Li
* May 11, 2021
*/

/* Function: Cluster the data in sample level*/

public class microaggregation {
	
	public static ArrayList<ArrayList<Record>> MDAV(ArrayList<Record> allrecord, int k) throws IOException{
		ArrayList<Record> input = allrecord;
		build(input);
		ArrayList<Record> MAXpts = new  ArrayList<Record>();
		ArrayList<ArrayList<Record>> result = new ArrayList<ArrayList<Record>>();
		while(input.size() > 3*k-1){               						// number of data > 3k-1
			MAXpts = getMAXDIST(input);
			for(int i=0; i<MAXpts.size();i++){
				ArrayList<Record> temp =getKNBR(input,MAXpts.get(i).getData(),k);
				result.add(temp);
				input.removeAll(temp);
			}
		}
		
		if(input.size() >= 2*k && input.size() <= 3*k-1 ){   			// 2k < remaining data <3k-1
			MAXpts = getMAXDIST(input);
			ArrayList<Record> temp = new ArrayList<Record>();
			temp = getKNBR(input,MAXpts.get(0).getData(),k);
			result.add(temp);
			input.removeAll(temp);

		}
		if(input.size() <= 2*k){                                      	//2k < remaining data <3k-1
			ArrayList<Record> temp = new ArrayList<Record>();	
			for(int i=0; i<input.size();i++){			
			temp.add(input.get(i));
			}
			result.add(temp);
		}
		
	return 	result;
		
	}
	
	public static ArrayList<Record> readfile(String input_filepath) throws IOException{ 			//read label
		String line="";
		int pos =0;
		ArrayList<Record> samplebag = new ArrayList<Record>();
		BufferedReader br = new BufferedReader(new FileReader(input_filepath));
		while((line = br.readLine())!=null){
			String[] temp = line.split(",");
			double[] sample = new double[temp.length];
			
			for(int i = 0; i<sample.length;i++){
				sample[i] =  Double.parseDouble(temp[i]);				
			}
			
			Record r = new Record(sample,pos);
			pos++;
			samplebag.add(r);
		}
		return samplebag;
	} 
	
	
	public static ArrayList<Record> build(ArrayList<Record> input) throws IOException{
		ArrayList<Record> samplebag = new ArrayList<Record>();
		for(int i=0;i<input.size();i++){
			input.get(i).setPos(i);			
		}
		return samplebag;
	}
	
	
	public static ArrayList<Record> getMAXDIST(ArrayList<Record> input) throws IOException{
		double[] mean = getmean(input);
		double dist_1 = Double.MAX_VALUE;
		double dist_2 = Double.MAX_VALUE;
		int index = 0;
		ArrayList<Record> mostdistpt = new ArrayList<Record>();
		
		for(int i=0;i<input.size();i++){
			dist_2 = getDIST(input.get(i).getData(),mean);
			if(dist_2 <dist_1){
				dist_1=dist_2;
				index=i;
			}
		}
		
		mostdistpt.add(input.get(index));
		double[] MAXDISTPoint = input.get(index).getData();
		dist_1=0;
		
		for(int j=0;j<input.size();j++){
			dist_2 = getDIST(input.get(j).getData(),MAXDISTPoint);
			if(dist_2 <dist_1){
				dist_1=dist_2;
				index=j;
			}
		}
		mostdistpt.add(input.get(index));
		return mostdistpt;
	}

	
	public static ArrayList<Record> getKNBR(ArrayList<Record> input, double[] center,int k) throws IOException{
		if(k<1 || k>input.size())
		System.out.println("Wrong k");		
		double mindist=Double.MAX_VALUE;
		int index =0;
		int[] indexarray= new int[k];
		double[] allDIST = new double[input.size()]; 
		ArrayList<Record> KNBR = new ArrayList<Record>();
		HashMap<Integer,Double> Map = new HashMap<Integer,Double>();
		for(int i=0; i<input.size(); i++ ){
			allDIST[i]=getDIST(center,input.get(i).getData());
			
			if(Map.containsKey(i))
				System.out.println("Map key confliction");
				
			Map.put(i,getDIST(center,input.get(i).getData()));
		}
		
		int n=k;
		while(n!=0){
			double tmp1 = mindist;
			index=0;
			for(int i=0;i<input.size(); i++){		
				double tmp2 = Map.get(i);
				if(tmp2<tmp1){
					tmp1=tmp2;
					index=i;	
				}
			}
			indexarray[n-1]=index;
			Map.put(index,Double.MAX_VALUE);
			n--;
		}
		
		Arrays.sort(allDIST);
		
		for(int j=0; j<k;j++ ){
			KNBR.add(input.get(indexarray[j]));
		}

		return KNBR;
	}
		
	public static double[] datatransfer(String[] StringArray) {
		double[] doublearray= new double[StringArray.length];
		doublearray[0] = Double.parseDouble(StringArray[0]);
		for(int i=1;i<StringArray.length;i++){
			String[] temp = StringArray[i].split(":");
			doublearray[i] = Double.parseDouble(temp[1]);
		}
		return doublearray;		
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
		//System.out.println("mean vector:"+Arrays.toString(mean));
		return mean;		
	}
	
	public static double getDIST(double[] a,double[] b) throws IOException{
		double dist = Double.MIN_VALUE; 
		for(int i=0;i<a.length;i++){
			double tmp = Math.pow(a[i], 2) + Math.pow(b[i], 2);
			dist+=Math.sqrt(tmp);
		}
		return dist;
	}	
	
	
}
