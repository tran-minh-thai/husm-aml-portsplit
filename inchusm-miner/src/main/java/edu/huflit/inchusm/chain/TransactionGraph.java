package edu.huflit.inchusm.chain;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoublePredicate;

/**
 * Directed transaction multigraph: accounts carry a label, every transaction carries a source,
 * a target, a label and a positive amount. Several transactions may join the same two accounts.
 *
 * <p>Transactions out of (and into) every account are kept sorted by (label, amount, id), so the
 * transactions that conserve the amount of a given transaction are one contiguous range found by
 * binary search on the conservation predicate itself (no division, no tolerance).
 *
 * <p>Self-transactions (source = target) can never be part of a chain occurrence, whose accounts
 * are pairwise distinct; they are dropped at construction and counted.
 *
 * <p>File format ({@code .tg}), one record per line, {@code #} starts a comment line:
 * <pre>
 *   v ID LABEL
 *   e SOURCE TARGET LABEL AMOUNT
 * </pre>
 * Account ids need not be consecutive; they are remapped to 0..n-1 in order of appearance.
 */
public final class TransactionGraph {

    private final int[] vlabel;
    private final int[] src;
    private final int[] dst;
    private final int[] elabel;
    private final double[] amount;
    private final int[] outStart;
    private final int[] outTx;
    private final int[] inStart;
    private final int[] inTx;
    private final int droppedSelfTransactions;

    public TransactionGraph(int[] vlabel, int[] src, int[] dst, int[] elabel, double[] amount) {
        int m = src.length;
        if (dst.length != m || elabel.length != m || amount.length != m) {
            throw new IllegalArgumentException("transaction arrays differ in length");
        }
        this.vlabel = vlabel.clone();
        for (int i = 0; i < this.vlabel.length; i++) {
            if (this.vlabel[i] < 0) throw new IllegalArgumentException("account " + i + " has a negative label");
        }
        int kept = 0;
        for (int i = 0; i < m; i++) {
            checkTransaction(i, src[i], dst[i], elabel[i], amount[i]);
            if (src[i] != dst[i]) kept++;
        }
        this.src = new int[kept];
        this.dst = new int[kept];
        this.elabel = new int[kept];
        this.amount = new double[kept];
        int j = 0;
        for (int i = 0; i < m; i++) {
            if (src[i] == dst[i]) continue;
            this.src[j] = src[i];
            this.dst[j] = dst[i];
            this.elabel[j] = elabel[i];
            this.amount[j] = amount[i];
            j++;
        }
        this.droppedSelfTransactions = m - kept;
        int n = this.vlabel.length;
        outStart = new int[n + 1];
        inStart = new int[n + 1];
        outTx = csr(this.src, outStart, n);
        inTx = csr(this.dst, inStart, n);
    }

    private void checkTransaction(int i, int s, int t, int lab, double a) {
        int n = vlabel.length;
        if (s < 0 || s >= n || t < 0 || t >= n) {
            throw new IllegalArgumentException("transaction " + i + " names an account outside 0.." + (n - 1));
        }
        if (lab < 0) throw new IllegalArgumentException("transaction " + i + " has a negative label");
        if (!(a > 0) || Double.isInfinite(a)) {
            throw new IllegalArgumentException("transaction " + i + " has amount " + a
                    + "; amounts must be positive and finite (the conservation ratio is undefined otherwise)");
        }
    }

    /** Transaction ids grouped by account, each group sorted by (label, amount, id). */
    private int[] csr(int[] key, int[] start, int n) {
        int m = key.length;
        for (int e = 0; e < m; e++) start[key[e] + 1]++;
        for (int v = 0; v < n; v++) start[v + 1] += start[v];
        Integer[] order = new Integer[m];
        for (int e = 0; e < m; e++) order[e] = e;
        Arrays.sort(order, (x, y) -> {
            if (key[x] != key[y]) return Integer.compare(key[x], key[y]);
            if (elabel[x] != elabel[y]) return Integer.compare(elabel[x], elabel[y]);
            int c = Double.compare(amount[x], amount[y]);
            return c != 0 ? c : Integer.compare(x, y);
        });
        int[] out = new int[m];
        for (int i = 0; i < m; i++) out[i] = order[i];
        return out;
    }

    public static TransactionGraph load(Path path) throws IOException {
        Map<Long, Integer> idMap = new HashMap<>();
        List<Integer> labels = new ArrayList<>();
        List<long[]> rawEdges = new ArrayList<>();
        List<Integer> edgeLabels = new ArrayList<>();
        List<Double> amounts = new ArrayList<>();
        int lineNo = 0;
        for (String raw : Files.readAllLines(path)) {
            lineNo++;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] tok = line.split("\\s+");
            String where = path.getFileName() + ":" + lineNo + ": ";
            switch (tok[0]) {
                case "v" -> {
                    if (tok.length != 3) throw new IOException(where + "an account line needs 'v ID LABEL'");
                    long id = Long.parseLong(tok[1]);
                    if (idMap.putIfAbsent(id, labels.size()) != null) {
                        throw new IOException(where + "account " + id + " is declared twice");
                    }
                    labels.add(Integer.parseInt(tok[2]));
                }
                case "e" -> {
                    if (tok.length != 5) {
                        throw new IOException(where + "a transaction line needs 'e SOURCE TARGET LABEL AMOUNT'");
                    }
                    rawEdges.add(new long[] {Long.parseLong(tok[1]), Long.parseLong(tok[2])});
                    edgeLabels.add(Integer.parseInt(tok[3]));
                    amounts.add(Double.parseDouble(tok[4]));
                }
                default -> throw new IOException(where + "unknown record '" + tok[0] + "'");
            }
        }
        int m = rawEdges.size();
        int[] s = new int[m];
        int[] t = new int[m];
        int[] lab = new int[m];
        double[] a = new double[m];
        for (int i = 0; i < m; i++) {
            Integer u = idMap.get(rawEdges.get(i)[0]);
            Integer v = idMap.get(rawEdges.get(i)[1]);
            if (u == null || v == null) {
                throw new IOException(path.getFileName() + ": transaction " + i + " names an undeclared account");
            }
            s[i] = u;
            t[i] = v;
            lab[i] = edgeLabels.get(i);
            a[i] = amounts.get(i);
        }
        int[] vl = new int[labels.size()];
        for (int i = 0; i < vl.length; i++) vl[i] = labels.get(i);
        return new TransactionGraph(vl, s, t, lab, a);
    }

    public int numAccounts() {
        return vlabel.length;
    }

    public int numTransactions() {
        return src.length;
    }

    public int droppedSelfTransactions() {
        return droppedSelfTransactions;
    }

    public int accountLabel(int v) {
        return vlabel[v];
    }

    public int source(int e) {
        return src[e];
    }

    public int target(int e) {
        return dst[e];
    }

    public int label(int e) {
        return elabel[e];
    }

    public double amount(int e) {
        return amount[e];
    }

    /** Slot range [from, to) of {@link #outSlot} for transactions out of v with label lab. */
    public int[] outByLabel(int v, int lab) {
        return labelRange(outTx, outStart[v], outStart[v + 1], lab);
    }

    /** Slot range [from, to) of {@link #inSlot} for transactions into v with label lab. */
    public int[] inByLabel(int v, int lab) {
        return labelRange(inTx, inStart[v], inStart[v + 1], lab);
    }

    public int outSlot(int i) {
        return outTx[i];
    }

    public int inSlot(int i) {
        return inTx[i];
    }

    /**
     * Narrow a label range of out-slots (or in-slots) to the amounts satisfying {@code keep}, which
     * must be monotone along the sorted amounts in the sense that the satisfying amounts form one
     * interval: {@code atLeast} is false-then-true, {@code atMost} is true-then-false.
     */
    public int[] narrow(boolean out, int[] range, DoublePredicate atLeast, DoublePredicate atMost) {
        int[] slots = out ? outTx : inTx;
        int lo = range[0];
        int hi = range[1];
        int a = lo;
        int b = hi;
        while (a < b) {                       // first slot whose amount satisfies atLeast
            int mid = (a + b) >>> 1;
            if (atLeast.test(amount[slots[mid]])) b = mid;
            else a = mid + 1;
        }
        int from = a;
        a = from;
        b = hi;
        while (a < b) {                       // first slot whose amount fails atMost
            int mid = (a + b) >>> 1;
            if (atMost.test(amount[slots[mid]])) a = mid + 1;
            else b = mid;
        }
        return new int[] {from, a};
    }

    private int[] labelRange(int[] slots, int lo, int hi, int lab) {
        int a = lo;
        int b = hi;
        while (a < b) {
            int mid = (a + b) >>> 1;
            if (elabel[slots[mid]] < lab) a = mid + 1;
            else b = mid;
        }
        int from = a;
        b = hi;
        while (a < b) {
            int mid = (a + b) >>> 1;
            if (elabel[slots[mid]] <= lab) a = mid + 1;
            else b = mid;
        }
        return new int[] {from, a};
    }
}
