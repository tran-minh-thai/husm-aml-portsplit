package edu.huflit.inchusm.chain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Mines every conserved flow chain whose minimum-image support reaches a threshold.
 *
 * <p><b>Occurrence.</b> An occurrence of a chain pattern of length m is an assignment of distinct
 * accounts to its positions together with one transaction per pattern edge, with the labels of the
 * pattern, such that at every intermediate position i (1..m-1) the transaction that leaves keeps
 * the amount of the one that arrives: {@code (1-theta) a_in <= a_out <= (1+theta) a_in}.
 *
 * <p><b>Support.</b> The image set of a position is the set of accounts it takes over all valid
 * occurrences; the support is the size of the smallest image set (minimum image based support).
 *
 * <p><b>Why the search is exact and level-wise.</b> Restricting a valid occurrence of an open chain
 * to its first m-1 transactions, or to its last m-1, gives a valid occurrence of the prefix
 * (resp. suffix) chain with the same accounts at the shared positions: the conservation conditions
 * kept are a subset of the original ones. Hence the image sets of a chain are contained, position
 * by position, in those of its prefix and of its suffix, and its support is at most theirs. The
 * same holds for a closed chain with respect to the two open chains obtained by removing its first
 * or its last transaction, because the only pair a closed chain does not constrain is the pair at
 * its origin, and removing either of those transactions removes the origin as an intermediate
 * position. So every chain of length m+1 that reaches the threshold is the join of two chains of
 * length m that reach it (prefix and suffix), and its image sets lie in the intersection of theirs.
 * The miner generates exactly these joins and decides every remaining account by a search pinned to
 * that account, so the reported image sets and supports are exact.
 */
public final class ChainMiner {

    public record Params(double theta, int minSupport, int maxLength, int minCycleLength, boolean closedChains) {
        public Params {
            if (!(theta >= 0 && (theta < 1 || theta == Double.POSITIVE_INFINITY))) {
                throw new IllegalArgumentException("theta must lie in [0, 1), or be infinite for no conservation");
            }
            if (minSupport < 1) throw new IllegalArgumentException("minimum support must be at least 1");
            if (maxLength < 1) throw new IllegalArgumentException("maximum length must be at least 1");
            if (minCycleLength < 2) throw new IllegalArgumentException("a closed chain has at least 2 transactions");
        }
    }

    /**
     * One reported chain. markedCounts (accounts of each image set that are marked) is null unless
     * marked accounts were given; images (account indices per position) is null unless kept.
     */
    public record Result(ChainPattern pattern, int support, int[] imageCounts, int[] markedCounts, int[][] images) {
    }

    /** Machine-independent work counters. */
    public static final class Counters {
        public long candidates;
        public long prunedByDomain;
        public long prunedByConsistency;
        public long localRejections;
        public long budgetFallbacks;
        public long searches;
        public long searchesFound;
        public long searchesFailed;
        public long earlyStops;
        public final Map<Integer, long[]> frequentByLength = new TreeMap<>();   // length -> {open, closed}

        void frequent(ChainPattern p) {
            frequentByLength.computeIfAbsent(p.length(), k -> new long[2])[p.closed() ? 1 : 0]++;
        }
    }

    /** slots (consistent slots per pattern edge 1..m) is kept only by the slot probe. */
    private record Entry(ChainPattern pattern, int[][] images, int[][] slots) {
    }

    /**
     * Slot probe, per candidate length: candidates that reached slot consistency, the total size of
     * the slots built from the domains, of the slots inherited from prefix and suffix (intersection,
     * restricted to the domains), of the consistent slots, consistent-slot transactions outside the
     * inherited slots (zero if inheritance is sound), and the total slot size kept by the frequent open
     * chains of that length. Mining is unchanged by the probe.
     */
    public static final class SlotProbe {
        public final Map<Integer, long[]> byLength = new TreeMap<>();

        long[] at(int length) {
            return byLength.computeIfAbsent(length, k -> new long[6]);
        }
    }

    /** Steps a depth-first attempt on a closed chain may take before local propagation is used. */
    static final long DFS_BUDGET = 10_000;
    private long dfsBudget = DFS_BUDGET;
    private boolean useConsistency = true;
    private int consistencyFrom = 2;

    /** The conservation predicate between an arriving amount and the amount that leaves. */
    public static boolean conserves(double in, double out, double theta) {
        return out >= (1 - theta) * in && out <= (1 + theta) * in;
    }

    private final TransactionGraph g;
    private final Params params;
    private final Counters counters = new Counters();
    private boolean[] marked;
    private boolean keepImages;
    private SlotProbe slotProbe;
    private final Map<String, BitSet> accounts = new TreeMap<>();

    public ChainMiner(TransactionGraph g, Params params) {
        this.g = g;
        this.params = params;
    }

    /** Accounts to count in every image set (for example accounts known to be laundering); mining ignores them. */
    public ChainMiner marked(boolean[] marked) {
        if (marked.length != g.numAccounts()) throw new IllegalArgumentException("one flag per account is needed");
        this.marked = marked.clone();
        return this;
    }

    /** Switch slot consistency off (plain pinned search), to measure what it contributes; output is the same. */
    public ChainMiner consistency(boolean on) {
        this.useConsistency = on;
        return this;
    }

    /** Apply slot consistency only to candidates of at least this many transactions (output is the same). */
    public ChainMiner consistencyFrom(int length) {
        if (length < 2) throw new IllegalArgumentException("candidates have at least 2 transactions");
        this.consistencyFrom = length;
        return this;
    }

    /** Budget of the first depth-first attempt on a closed chain; only speed depends on it (tests use 0). */
    ChainMiner dfsBudget(long steps) {
        this.dfsBudget = steps;
        return this;
    }

    /** Record the slot probe (keeps the consistent slots of every open chain of the current level). */
    public ChainMiner slotProbe(boolean on) {
        this.slotProbe = on ? new SlotProbe() : null;
        return this;
    }

    public SlotProbe slotProbeStats() {
        return slotProbe;
    }

    /** Keep the image sets themselves in the results (memory grows with every reported chain). */
    public ChainMiner keepImages(boolean keep) {
        this.keepImages = keep;
        return this;
    }

    /**
     * After {@link #mine()}: for every kind and length, the accounts that occupy an intermediate
     * position (1..m-1) of some reported chain, and for closed chains the accounts at the origin.
     * Keys look like {@code open 3 intermediate} or {@code closed 4 origin}.
     */
    public Map<String, BitSet> accountsByRole() {
        return accounts;
    }

    public Counters counters() {
        return counters;
    }

    public List<Result> mine() {
        List<Result> results = new ArrayList<>();
        List<Entry> level = levelOne();
        for (Entry e : level) results.add(result(e));
        for (int k = 1; k < params.maxLength() && !level.isEmpty(); k++) {
            Map<String, List<Entry>> byPrefix = new HashMap<>();
            for (Entry q : level) byPrefix.computeIfAbsent(prefixKey(q.pattern()), x -> new ArrayList<>()).add(q);
            List<Entry> next = new ArrayList<>();
            for (Entry p : level) {
                List<Entry> qs = byPrefix.get(suffixKey(p.pattern()));
                if (qs == null) continue;
                for (Entry q : qs) {
                    int[] vlab = Arrays.copyOf(p.pattern().vlab(), k + 2);
                    vlab[k + 1] = q.pattern().vlab()[k];
                    int[] elab = Arrays.copyOf(p.pattern().elab(), k + 1);
                    elab[k] = q.pattern().elab()[k - 1];
                    Entry open = evaluate(new ChainPattern(vlab, elab, false), p, q);
                    if (open != null) {
                        next.add(open);
                        results.add(result(open));
                    }
                    if (params.closedChains() && k + 1 >= params.minCycleLength() && vlab[0] == vlab[k + 1]) {
                        Entry closed = evaluate(new ChainPattern(vlab, elab, true), p, q);
                        if (closed != null) results.add(result(closed));   // a closed chain extends no further
                    }
                }
            }
            level = next;
        }
        results.sort(Comparator.comparing((Result r) -> r.pattern().length())
                .thenComparing(r -> r.pattern().key()));
        return results;
    }

    private Result result(Entry e) {
        int[][] img = e.images();
        int[] counts = new int[img.length];
        int[] markedCounts = marked == null ? null : new int[img.length];
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < counts.length; i++) {
            counts[i] = img[i].length;
            min = Math.min(min, counts[i]);
            if (markedCounts != null) {
                for (int v : img[i]) if (marked[v]) markedCounts[i]++;
            }
        }
        ChainPattern p = e.pattern();
        String kind = p.closed() ? "closed " : "open ";
        if (p.length() >= 2) {
            BitSet mid = accounts.computeIfAbsent(kind + p.length() + " intermediate", k -> new BitSet());
            for (int i = 1; i < p.length(); i++) for (int v : img[i]) mid.set(v);
        }
        if (p.closed()) {
            BitSet origin = accounts.computeIfAbsent(kind + p.length() + " origin", k -> new BitSet());
            for (int v : img[0]) origin.set(v);
        }
        counters.frequent(p);
        return new Result(p, min, counts, markedCounts, keepImages ? img : null);
    }

    // ---------------------------------------------------------------- length one

    private List<Entry> levelOne() {
        Map<String, int[]> labels = new TreeMap<>();
        Map<String, List<int[]>> ends = new HashMap<>();
        for (int e = 0; e < g.numTransactions(); e++) {
            int s = g.source(e);
            int t = g.target(e);
            int[] lab = {g.accountLabel(s), g.label(e), g.accountLabel(t)};
            String key = lab[0] + " " + lab[1] + " " + lab[2];
            labels.putIfAbsent(key, lab);
            ends.computeIfAbsent(key, x -> new ArrayList<>()).add(new int[] {s, t});
        }
        List<Entry> out = new ArrayList<>();
        for (Map.Entry<String, int[]> kv : labels.entrySet()) {
            counters.candidates++;
            List<int[]> list = ends.get(kv.getKey());
            int[] s = new int[list.size()];
            int[] t = new int[list.size()];
            for (int i = 0; i < s.length; i++) {
                s[i] = list.get(i)[0];
                t[i] = list.get(i)[1];
            }
            int[][] img = {distinctSorted(s), distinctSorted(t)};
            if (Math.min(img[0].length, img[1].length) < params.minSupport()) {
                counters.prunedByDomain++;
                continue;
            }
            int[] lab = kv.getValue();
            int[][] slots = null;
            if (slotProbe != null) {
                slots = new int[][] {null, Consistency.slot(g, img[0], lab[1], img[1])};
                slotProbe.at(1)[5] += slots[1].length;
            }
            out.add(new Entry(new ChainPattern(new int[] {lab[0], lab[2]}, new int[] {lab[1]}, false), img, slots));
        }
        return out;
    }

    // ---------------------------------------------------------------- join keys

    /** Open chain of length k: positions 0..k-1 and transactions 1..k-1. */
    private static String prefixKey(ChainPattern p) {
        return subKey(p, 0, p.length() - 1);
    }

    /** Open chain of length k: positions 1..k and transactions 2..k. */
    private static String suffixKey(ChainPattern p) {
        return subKey(p, 1, p.length() - 1);
    }

    private static String subKey(ChainPattern p, int first, int edges) {
        StringBuilder sb = new StringBuilder().append(p.vlab()[first]);
        for (int i = 0; i < edges; i++) {
            sb.append(" >").append(p.elab()[first + i]).append(' ').append(p.vlab()[first + i + 1]);
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- candidate evaluation

    /** p is the prefix and q the suffix of the candidate; both have length k. */
    private Entry evaluate(ChainPattern cand, Entry p, Entry q) {
        counters.candidates++;
        int k = p.pattern().length();
        int[][] pi = p.images();
        int[][] qi = q.images();
        int[][] dom = new int[cand.positions()][];
        if (cand.closed()) {
            dom[0] = intersect(pi[0], qi[k]);              // the suffix ends at the origin
        } else {
            dom[0] = pi[0];
            dom[k + 1] = qi[k];
        }
        for (int i = 1; i <= k; i++) dom[i] = intersect(pi[i], qi[i - 1]);
        for (int[] d : dom) {
            if (d.length < params.minSupport()) {
                counters.prunedByDomain++;
                return null;
            }
        }
        boolean filtered = useConsistency && cand.length() >= consistencyFrom;
        Search search = new Search(cand, dom, filtered);
        int[][] inherited = slotProbe != null && filtered && p.slots() != null && q.slots() != null
                ? inherited(cand, dom, p, q) : null;
        boolean consistent = true;
        if (!filtered) {
            search.plain();
        } else {
            consistent = search.consistent();
            if (inherited != null) probe(cand, inherited, search);
            if (!consistent) {
                counters.prunedByConsistency++;
                return null;
            }
        }
        int[][] img = search.images();
        if (img == null) return null;
        int[][] kept = null;
        if (inherited != null && !cand.closed()) {
            kept = search.slots.clone();
            long n = 0;
            for (int s = 1; s < kept.length; s++) n += kept[s].length;
            slotProbe.at(cand.length())[5] += n;
        }
        return new Entry(cand, img, kept);
    }

    /** Slots of the candidate inherited from its prefix p (edges 1..k) and suffix q (edges 2..k+1), within the domains. */
    private int[][] inherited(ChainPattern cand, int[][] dom, Entry p, Entry q) {
        int m = cand.length();
        int[][] ps = p.slots();
        int[][] qs = q.slots();
        int[][] in = new int[m + 1][];
        for (int s = 1; s <= m; s++) {
            int[] x = s == 1 ? ps[1] : s == m ? qs[m - 1] : intersect(ps[s], qs[s - 1]);
            int to = cand.closed() && s == m ? 0 : s;
            in[s] = Consistency.byEnd(g, Consistency.byEnd(g, x, dom[s - 1], true), dom[to], false);
        }
        return in;
    }

    private void probe(ChainPattern cand, int[][] inherited, Search search) {
        long[] a = slotProbe.at(cand.length());
        a[0]++;
        for (int s = 1; s < inherited.length; s++) {
            a[1] += search.fresh[s];
            a[2] += inherited[s].length;
            a[3] += search.slots[s].length;
            for (int e : search.slots[s]) if (Arrays.binarySearch(inherited[s], e) < 0) a[4]++;
        }
    }

    /** Decides, for every account of every position domain, whether it is in the image set. */
    private final class Search {
        private final int m;
        private final boolean closed;
        private final int[] elab;
        private int[][] dom;
        private BitSet[] rejected;
        private final int[][] slots;    // transactions allowed at each pattern edge 1..m after consistency
        private final int[] fresh;      // slot sizes built from the domains, before propagation (slot probe)
        private int[][] allowed;        // transactions allowed in the current pinned search
        private long steps;             // transactions tried in the current depth-first attempt
        private long budget;            // abandon the attempt (not the question) past this many
        private final int[] vert;       // account at each position, -1 if unassigned
        private final int[] tx;         // transaction at each pattern edge 1..m, -1 if unassigned

        private final boolean filtered;   // slot consistency applied to this candidate

        Search(ChainPattern cand, int[][] dom, boolean filtered) {
            this.filtered = filtered;
            this.m = cand.length();
            this.closed = cand.closed();
            this.elab = cand.elab();
            this.dom = dom;
            this.slots = new int[m + 1][];
            this.fresh = new int[m + 1];
            this.vert = new int[dom.length];
            this.tx = new int[m + 1];
        }

        /**
         * Reduce every slot to the transactions that lie on a complete conserving walk with the
         * pattern's labels and domains (and, for a closed chain, one that starts and ends at an
         * account of the origin domain), then shrink the domains to the accounts those transactions
         * use. False when a domain falls below the minimum support.
         */
        boolean consistent() {
            double theta = params.theta();
            for (int s = 1; s <= m; s++) {
                slots[s] = Consistency.slot(g, dom[s - 1], elab[s - 1], dom[pos(s)]);
                fresh[s] = slots[s].length;
            }
            while (true) {
                long before = size();
                for (int s = 2; s <= m; s++) slots[s] = Consistency.forward(g, slots[s - 1], slots[s], theta);
                for (int s = m - 1; s >= 1; s--) slots[s] = Consistency.backward(g, slots[s], slots[s + 1], theta);
                if (closed) {
                    int[] origins = intersect(Consistency.ends(g, slots[1], true), Consistency.ends(g, slots[m], false));
                    slots[1] = Consistency.byEnd(g, slots[1], origins, true);
                    slots[m] = Consistency.byEnd(g, slots[m], origins, false);
                }
                if (size() == before) break;
            }
            int[][] d = new int[dom.length][];
            d[0] = Consistency.ends(g, slots[1], true);
            for (int p = 1; p < dom.length; p++) d[p] = Consistency.ends(g, slots[p], false);
            for (int[] x : d) if (x.length < params.minSupport()) return false;
            dom = d;
            rejected = new BitSet[dom.length];
            for (int i = 0; i < dom.length; i++) rejected[i] = new BitSet(dom[i].length);
            return true;
        }

        /** Without consistency: domains as given, every transaction allowed. */
        void plain() {
            rejected = new BitSet[dom.length];
            for (int i = 0; i < dom.length; i++) rejected[i] = new BitSet(dom[i].length);
        }

        private long size() {
            long n = 0;
            for (int s = 1; s <= m; s++) n += slots[s].length;
            return n;
        }

        /**
         * For a closed chain pinned at account u: the transactions reachable from u going forward to
         * the origin and backward to it, each side conserving, kept only where both sides can meet
         * at one origin account. False when they cannot, which proves u is not in the image set.
         */
        private boolean localClosed(int pin, int u) {
            double theta = params.theta();
            int[][] a = new int[m + 1][];
            int[] at = {u};
            if (pin == 0) {
                a[1] = Consistency.byEnd(g, slots[1], at, true);
                for (int s = 2; s <= m; s++) a[s] = Consistency.expand(g, a[s - 1], elab[s - 1], slots[s], theta, true);
                a[m] = Consistency.byEnd(g, a[m], at, false);
                for (int s = m - 1; s >= 1; s--) a[s] = Consistency.backward(g, a[s], a[s + 1], theta);
                for (int s = 1; s <= m; s++) if (a[s].length == 0) return false;
                allowed = a;
                return true;
            }
            a[pin + 1] = Consistency.byEnd(g, slots[pin + 1], at, true);
            for (int s = pin + 2; s <= m; s++) a[s] = Consistency.expand(g, a[s - 1], elab[s - 1], slots[s], theta, true);
            a[pin] = Consistency.byEnd(g, slots[pin], at, false);
            for (int s = pin - 1; s >= 1; s--) a[s] = Consistency.expand(g, a[s + 1], elab[s - 1], slots[s], theta, false);
            while (true) {
                long before = 0;
                for (int s = 1; s <= m; s++) before += a[s].length;
                int[] origins = intersect(Consistency.ends(g, a[1], true), Consistency.ends(g, a[m], false));
                int self = Arrays.binarySearch(origins, u);
                if (self >= 0) {                                   // the origin is a different account
                    int[] o = new int[origins.length - 1];
                    System.arraycopy(origins, 0, o, 0, self);
                    System.arraycopy(origins, self + 1, o, self, origins.length - self - 1);
                    origins = o;
                }
                a[1] = Consistency.byEnd(g, a[1], origins, true);
                a[m] = Consistency.byEnd(g, a[m], origins, false);
                for (int s = 2; s <= m; s++) a[s] = Consistency.forward(g, a[s - 1], a[s], theta);
                for (int s = m - 1; s >= 1; s--) a[s] = Consistency.backward(g, a[s], a[s + 1], theta);
                long after = 0;
                for (int s = 1; s <= m; s++) {
                    if (a[s].length == 0) return false;
                    after += a[s].length;
                }
                if (after == before) break;
            }
            allowed = a;
            return true;
        }

        /** Exact image sets, or null as soon as one position cannot reach the minimum support. */
        int[][] images() {
            int n = dom.length;
            BitSet[] witnessed = new BitSet[n];
            int[] alive = new int[n];
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                witnessed[i] = new BitSet(dom[i].length);
                alive[i] = dom[i].length;
                order[i] = i;
            }
            Arrays.sort(order, Comparator.comparingInt((Integer i) -> dom[i].length).thenComparingInt(i -> i));
            for (int pos : order) {
                for (int idx = 0; idx < dom[pos].length; idx++) {
                    if (witnessed[pos].get(idx) || rejected[pos].get(idx)) continue;
                    counters.searches++;
                    if (pinned(pos, dom[pos][idx])) {
                        counters.searchesFound++;
                        for (int q = 0; q < n; q++) witnessed[q].set(Arrays.binarySearch(dom[q], vert[q]));
                    } else {
                        counters.searchesFailed++;
                        rejected[pos].set(idx);
                        if (--alive[pos] < params.minSupport()) {
                            counters.earlyStops++;
                            return null;
                        }
                    }
                }
            }
            int[][] img = new int[n][];
            for (int q = 0; q < n; q++) {
                img[q] = new int[witnessed[q].cardinality()];
                int j = 0;
                for (int idx = witnessed[q].nextSetBit(0); idx >= 0; idx = witnessed[q].nextSetBit(idx + 1)) {
                    img[q][j++] = dom[q][idx];
                }
            }
            return img;
        }

        /** Position of pattern vertex s (0..m): the origin when s = m in a closed chain. */
        private int pos(int s) {
            return closed && s == m ? 0 : s;
        }

        /** Is there a valid occurrence with account u at position pin? Leaves it in vert/tx if so. */
        boolean pinned(int pin, int u) {
            if (!filtered) {
                allowed = null;
                return attempt(pin, u, Long.MAX_VALUE);
            }
            allowed = slots;
            if (!closed) return attempt(pin, u, Long.MAX_VALUE);
            // A closed chain must come back to its origin, which slot consistency cannot see; a
            // depth-first attempt that runs out of budget is completed after local propagation.
            if (attempt(pin, u, dfsBudget)) return true;
            if (steps <= dfsBudget) return false;                  // the attempt was exhaustive
            counters.budgetFallbacks++;
            if (!localClosed(pin, u)) {
                counters.localRejections++;
                return false;
            }
            return attempt(pin, u, Long.MAX_VALUE);
        }

        /** Depth-first search for an occurrence through u at pin; gives up after budget steps. */
        private boolean attempt(int pin, int u, long budget) {
            this.budget = budget;
            this.steps = 0;
            Arrays.fill(vert, -1);
            Arrays.fill(tx, -1);
            vert[pin] = u;
            return right(pin + 1, pin);
        }

        /** Choose transaction s (from position s-1 to s) for s = pin+1 .. m, then go left. */
        private boolean right(int s, int pin) {
            if (s > m) return left(pin);
            final double theta = params.theta();
            int[] range = g.outByLabel(vert[s - 1], elab[s - 1]);
            if (s - 1 > pin) {                          // intermediate position whose arrival is known
                double a = g.amount(tx[s - 1]);
                double lo = (1 - theta) * a;
                double hi = (1 + theta) * a;
                range = g.narrow(true, range, b -> b >= lo, b -> b <= hi);
            }
            int tp = pos(s);
            for (int i = range[0]; i < range[1]; i++) {
                int e = g.outSlot(i);
                if (allowed != null && Arrays.binarySearch(allowed[s], e) < 0) continue;
                if (++steps > budget) return false;
                int t = g.target(e);
                boolean assign = vert[tp] < 0;
                if (assign ? !admissible(tp, t) : t != vert[tp]) continue;
                if (assign) vert[tp] = t;
                tx[s] = e;
                if (right(s + 1, pin)) return true;
                tx[s] = -1;
                if (assign) vert[tp] = -1;
            }
            return false;
        }

        /** Choose transaction s (from position s-1 to s) for s = pin .. 1. */
        private boolean left(int s) {
            if (s < 1) return true;
            final double theta = params.theta();
            int[] range = g.inByLabel(vert[s], elab[s - 1]);
            if (s <= m - 1 && tx[s + 1] >= 0) {        // position s is intermediate; what leaves it is known
                double b = g.amount(tx[s + 1]);
                range = g.narrow(false, range, a -> (1 + theta) * a >= b, a -> (1 - theta) * a <= b);
            }
            int fp = s - 1;
            for (int i = range[0]; i < range[1]; i++) {
                int e = g.inSlot(i);
                if (allowed != null && Arrays.binarySearch(allowed[s], e) < 0) continue;
                if (++steps > budget) return false;
                int f = g.source(e);
                boolean assign = vert[fp] < 0;
                if (assign ? !admissible(fp, f) : f != vert[fp]) continue;
                if (assign) vert[fp] = f;
                tx[s] = e;
                if (left(s - 1)) return true;
                tx[s] = -1;
                if (assign) vert[fp] = -1;
            }
            return false;
        }

        private boolean admissible(int p, int v) {
            int idx = Arrays.binarySearch(dom[p], v);
            if (idx < 0 || rejected[p].get(idx)) return false;
            for (int w : vert) {
                if (w == v) return false;
            }
            return true;
        }
    }

    // ---------------------------------------------------------------- sorted int sets

    static int[] distinctSorted(int[] a) {
        int[] s = a.clone();
        Arrays.sort(s);
        int j = 0;
        for (int i = 0; i < s.length; i++) {
            if (i == 0 || s[i] != s[i - 1]) s[j++] = s[i];
        }
        return Arrays.copyOf(s, j);
    }

    static int[] intersect(int[] a, int[] b) {
        int[] out = new int[Math.min(a.length, b.length)];
        int i = 0;
        int j = 0;
        int k = 0;
        while (i < a.length && j < b.length) {
            if (a[i] < b[j]) i++;
            else if (a[i] > b[j]) j++;
            else {
                out[k++] = a[i];
                i++;
                j++;
            }
        }
        return Arrays.copyOf(out, k);
    }
}
