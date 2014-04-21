import java.util.Arrays;

import il2.inf.bp.*;
import il2.inf.bp.schedules.MessagePassingScheduler;
import il2.model.Domain;
import il2.model.Table;
import il2.util.IntSet;
import il2.util.Pair;

/*
 * Using Hugin for variable-to-factor message computation, while
 * SS for factor-to-variable message computation.
 */

public class MixedHuginSSBP extends BeliefPropagation {
	public Table[] messageProducts; // normalized product of all msgs from factor to var
	
    public MixedHuginSSBP(Table[] tables, MessagePassingScheduler s,
                      int mi, long tm, double ct) {
        super(tables,s,mi,tm,ct);
        initializeMessageProducts();
    }
    
    public MixedHuginSSBP(Table[] tables, int mi, long tm, double ct) {
    	super(tables,mi,tm,ct);
    	initializeMessageProducts();
    }
    
    public Table[] getMessageProducts() {
    	return messageProducts;
    }
    
    public static Table unitTable(Domain d, int var) {
    	Table res = Table.varTable(d, var);
    	Arrays.fill(res.values(), 1);
    	return res;
    }
    
    // initializing the message product table 
    private void initializeMessageProducts() {
    	messageProducts = new Table[domain.size()];
    	for (int var = 0; var < domain.size(); var++) { 
    		messageProducts[var] = unitTable(domain,  var);
    		messageProducts[var].normalizeInPlace();
    	}
    }

    protected Table computeMessage(Pair pair) {
    	Table msg;
    	int var = scheduler.varOfPair(pair);
    	
    	// factor-to-variable
    	if (pair.s1 > pair.s2) {
    		Table[] tabs = collectTables(pair);
    		msg = unitTable(domain,var);
    		msg.multiplyAndProjectInto(tabs);
    		
    		// update messageProducts[varNode]
    		Table oldmsg = messages.get(pair);
    		messageProducts[var].divideRelevantInto(oldmsg.values());
    		messageProducts[var].multiplyInto(msg);
    		messageProducts[var].normalizeInPlace();    		
    	}
    	// variable-to-factor
    	else {
    		msg = messageProducts[var].copy();
    		msg.divideRelevantInto(messages.get(pair).values());
    	}
    	msg.normalizeInPlace();
		return msg;		
    }

    protected double computeResidual(Table oldt, Table newt) {
        double[] oldv = oldt.values();
        double[] newv = newt.values();
        double residual = 0.0;
        for (int i = 0; i < oldv.length; i++) {
            double delta = oldv[i]-newv[i];
            if ( delta < 0 ) delta = -delta;
            if ( delta > residual ) residual = delta;
        }
        return residual;
    }

    public double logPrEvidence() {
        makeValid();
        return 0.0;
    }
    public double prEvidence() {
        return Math.exp(logPrEvidence());
    }

    public Table tableConditional(int table) {
        makeValid();
        Table[] tabs = collectTables(table + domain.size());
        Table t = Table.createCompatible(tables[table]);
        t.multiplyAndProjectInto(tabs);
        t.normalizeInPlace();
        return t;
    }
    
    public Table tableJoint(int table) { return tableConditional(table); }
    
    public Table varConditional(int var) {
        makeValid();
        Table[] tabs = collectTables(var);
        Table t = Table.varTable(domain,var);
        t.multiplyAndProjectInto(tabs);
        t.normalizeInPlace(); 
       return t;
    }
    
    public Table varJoint(int var) { return varConditional(var); }

    public int[] mpe() {
        makeValid();
        int[] mpe = new int[domain.size()];
        for (int var = 0; var < mpe.length; var++) {
            mpe[var] = varConditional(var).maxIndex();
        }
        return mpe;
    }
    
    /*
    private int findLargestMsgSize(){
		int result=0;
		for(int i=0;i<messageProducts.length;i++){
		    if(messageProducts[i].sizeInt()>result){
			result=messageProducts[i].sizeInt();
		    }
		}
		return result;
    }
    */
    /**
     * check MPE assignment against evidence
     */
    protected boolean isInvalidWorld(int[] mpe) {
        for (int i = 0; i < evidence.size(); i++) {
            int var = evidence.keys().get(i);
            int state = evidence.get(var);
            if ( mpe[var] != state )
                return true;            
        }
        return false;
    }

    public double logPrMPE() {
        int[] mpe = mpe();
        return logPrMPE(tables,mpe);
    }

    public double logPrMPE(int[] mpe) {
        if ( isInvalidWorld(mpe) )
            return Double.NEGATIVE_INFINITY;
        else
            return logPrMPE(tables,mpe);
    }

    protected static double logPrMPE(Table[] tables, int[] mpe) {
        double log_mpe = 0.0;
        for ( Table table : tables ) {
            IntSet vars = table.vars();
            int[] inst = new int[vars.size()];
            for (int i = 0; i < vars.size(); i++) {
                int var = vars.get(i);
                inst[i] = mpe[var];
            }
            int index = table.getIndexFromFullInstance(inst);
            log_mpe += Math.log( table.values()[index] );
        }
        return log_mpe;
    }

    public double edgeScore(int[] edge) {
        int n = domain.size();
        Pair p1 = new Pair(edge[0],edge[1]+n);
        Pair p2 = new Pair(edge[1]+n,edge[0]);
        double score;
        if ( isUnitTablePair(p1) ) {
            // note that in this case, this edge will always be part
            // of the spanning tree
            score = 1.0;
        } else {
            double[] m1 = messages.get(p1).values();
            double[] m2 = messages.get(p2).values();

            double gamma1 = m1[0]*m2[0];
            for (int i = 1; i < m1.length; i++) {
                double cur = m1[i]*m2[i];
                if ( cur > gamma1 ) gamma1 = cur;
            }
            double gamma2 = m1[0], gamma3 = m2[0];
            for (int i = 1; i < m1.length; i++) {
                if ( m1[i] > gamma2 ) gamma2 = m1[i];
                if ( m2[i] > gamma3 ) gamma3 = m2[i];
            }
            score = gamma2*gamma3/gamma1;
        }
        return score;
    }
}