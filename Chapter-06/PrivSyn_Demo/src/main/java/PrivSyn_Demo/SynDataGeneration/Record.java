package PrivSyn_Demo.SynDataGeneration;
import java.util.*;

/**
 * Represents a record of the dataset.
 */
/** 
* @author Mingchen Li
* May 11, 2021
*/
public class Record {
	
	double[] data;
	int pos, Group;
	
	/**
     * Default constructor.
     * @param size the number of attributes.
     */
	public Record(int size) {
		data = new double[size];
		for (int i=0; i<size; i++) 
			data[i]  = 0.0;
	}
	
	
	/**
     * Constructor.
     * @param data the values of the record.
     */
	public Record(double[] data) {
		this.data = data;
	}
	
	public Record(double[] data,int pos) {
		this.data = data;
		this.pos =pos;
	}
	
	public Record(Record rec) {
		data = new double[rec.getData().length];
		for (int i = 0; i< data.length; i++) {
			double[] tmp = rec.getData();
			data[i] = tmp[i];
		}
		this.pos = rec.getPos();
		this.Group = rec.getGroup();
	}
 
    
    
    /**
     * Returns all the values of the record.
     * @return the list of the values of the record.
     */
    
	public void setData (double[] data) {
    	this.data = data; 
    }

	
	public double[] getData(){
        return data;
    }
    
    
    /**
     * Returns a string representation of the record.
     * @return a string representation of the record.
     */
    public String toString() {
    	String s = "";
    	int i=0;
    	for (Double value: data) {
    		s+=value.toString();    		
    		i++;
    		if (i<data.length) s+=", ";
    	}
    	return s;
    }
   
    public int length() {
		return data.length;
	}


	public int getPos() {
		return this.pos;
	}
    
    public void setPos (int position) {
    	this.pos = position; 
    }


	public void setGroup(int group) {
		this.Group = group;
	}
	
	public int getGroup () {
		return this.Group;
	}

}
