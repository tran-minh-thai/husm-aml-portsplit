package edu.huflit.inchusm.cycles;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

/**
 * The accounts that lie on a simple directed cycle of exactly m transactions, for every m in a range.
 *
 * <p>With one account label and one transaction label and no amount constraint, every rotation of an
 * occurrence of the closed chain of length m is again an occurrence, so all image sets of that chain are
 * the same set C_m: the accounts on some cycle v_0 -> v_1 -> ... -> v_{m-1} -> v_0 with pairwise distinct
 * accounts. Its support is |C_m|. This program computes C_m directly, without mining the open chains.
 * It reads the graph on its own (it shares no code with the chain miner, so the two can be compared).
 *
 * <p>Parallel transactions are merged and self-transactions dropped (neither changes which accounts lie on
 * a simple cycle of length at least 3). Only edges inside one strongly connected component can lie on a
 * cycle, so the others are removed first. For every remaining account v not yet known to be in C_m, a
 * search decides whether some cycle of length m passes through v: the last one or two transactions
 * into v are tabulated, and paths leaving v are joined with them, keeping enough distinct second-to-last
 * accounts that one of them avoids the accounts of the leaving path. Every account of a cycle found is
 * added to C_m, and every cycle found is checked edge by edge before it is used. The search runs on the
 * graph or on its reverse, whichever is predicted to be cheaper (a cycle through v in one is a cycle
 * through v in the other); the result does not depend on that choice nor on the number of threads.
 *
 * <pre>
 * java -cp inchusm-miner.jar edu.huflit.inchusm.cycles.CycleMembership --graph g.tg --min-length 3
 *      --max-length 5 [--min-support 10] [--threads N] [--orientation auto|forward|reverse] --out accounts.tsv
 * </pre>
 * The output has the columns of the chain miner's --accounts-out (kind, length, role, account), with the
 * rows "closed m origin" and "closed m intermediate" for every m whose support reaches --min-support,
 * which are the accounts of the reported closed chains of length m.
 */
public final class CycleMembership {

    final int n;
    final long[] originalId;
    final int[] outOff, outAdj, inOff, inAdj;     // the graph restricted to strongly connected components
    final int distinctPairs, sccEdges, sccAccounts;
    final String orientation;

    final LongAdder searches = new LongAdder(), found = new LongAdder(), examined = new LongAdder();

    CycleMembership(long[] originalId, long[] pairs, String orientation) {
        this.n = originalId.length;
        this.originalId = originalId;
        this.orientation = orientation;
        this.distinctPairs = pairs.length;
        int[] off = offsets(pairs, true);
        int[] adj = new int[pairs.length];
        for (int i = 0; i < pairs.length; i++) adj[i] = (int) pairs[i];
        int[] comp = components(off, adj);
        int[] compSize = new int[n];
        for (int v = 0; v < n; v++) compSize[comp[v]]++;
        int kept = 0;
        for (long p : pairs) if (comp[(int) (p >>> 32)] == comp[(int) p]) kept++;
        long[] inside = new long[kept];
        int k = 0;
        for (long p : pairs) if (comp[(int) (p >>> 32)] == comp[(int) p]) inside[k++] = p;
        this.sccEdges = kept;
        int accounts = 0;
        for (int v = 0; v < n; v++) if (compSize[comp[v]] >= 2) accounts++;
        this.sccAccounts = accounts;
        outOff = offsets(inside, true);
        outAdj = new int[kept];
        for (int i = 0; i < kept; i++) outAdj[i] = (int) inside[i];      // sorted by source, then target
        inOff = offsets(inside, false);
        inAdj = new int[kept];
        int[] fill = Arrays.copyOf(inOff, n);
        for (long p : inside) inAdj[fill[(int) p]++] = (int) (p >>> 32);  // sources ascending within a target
    }

    /** CSR offsets of packed (source << 32 | target) pairs, by source or by target. */
    private int[] offsets(long[] pairs, boolean bySource) {
        int[] off = new int[n + 1];
        for (long p : pairs) off[(bySource ? (int) (p >>> 32) : (int) p) + 1]++;
        for (int v = 0; v < n; v++) off[v + 1] += off[v];
        return off;
    }

    /** Strongly connected components (iterative Tarjan); component ids are arbitrary but equal within one. */
    private int[] components(int[] off, int[] adj) {
        int[] index = new int[n], low = new int[n], comp = new int[n], stack = new int[n], callV = new int[n],
                callE = new int[n];
        boolean[] onStack = new boolean[n];
        Arrays.fill(index, -1);
        int counter = 0, sp = 0, comps = 0;
        for (int root = 0; root < n; root++) {
            if (index[root] >= 0) continue;
            int depth = 0;
            callV[0] = root;
            callE[0] = off[root];
            index[root] = low[root] = counter++;
            stack[sp++] = root;
            onStack[root] = true;
            while (depth >= 0) {
                int v = callV[depth];
                if (callE[depth] < off[v + 1]) {
                    int w = adj[callE[depth]++];
                    if (index[w] < 0) {
                        index[w] = low[w] = counter++;
                        stack[sp++] = w;
                        onStack[w] = true;
                        depth++;
                        callV[depth] = w;
                        callE[depth] = off[w];
                    } else if (onStack[w]) {
                        low[v] = Math.min(low[v], index[w]);
                    }
                    continue;
                }
                if (low[v] == index[v]) {
                    int w;
                    do {
                        w = stack[--sp];
                        onStack[w] = false;
                        comp[w] = comps;
                    } while (w != v);
                    comps++;
                }
                depth--;
                if (depth >= 0) low[callV[depth]] = Math.min(low[callV[depth]], low[v]);
            }
        }
        return comp;
    }

    /** C_m as a bit set over account indices. */
    AtomicLongArray members(int m, int threads) {
        if (m < 3) throw new IllegalArgumentException("cycle length must be at least 3");
        AtomicLongArray in = new AtomicLongArray((n + 63) / 64);
        Runnable work = () -> IntStream.range(0, n).parallel().forEach(v -> {
            if (outOff[v + 1] == outOff[v] || isSet(in, v)) return;
            int[] cycle = Scratch.get(n).search(this, v, m);
            searches.increment();
            if (cycle == null) return;
            check(cycle, m);
            found.increment();
            for (int a : cycle) set(in, a);
        });
        java.util.concurrent.ForkJoinPool pool = new java.util.concurrent.ForkJoinPool(threads);
        try {
            pool.submit(work).get();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        } finally {
            pool.shutdown();
        }
        return in;
    }

    /** A cycle used as a witness must be one: m distinct accounts, each transaction present. */
    void check(int[] cycle, int m) {
        if (cycle.length != m) throw new IllegalStateException("witness of length " + cycle.length + ", expected " + m);
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                if (cycle[i] == cycle[j]) throw new IllegalStateException("witness repeats an account: " + Arrays.toString(cycle));
            }
            if (!contains(outAdj, outOff[cycle[i]], outOff[cycle[i] + 1], cycle[(i + 1) % m])) {
                throw new IllegalStateException("witness uses a missing transaction: " + Arrays.toString(cycle));
            }
        }
    }

    static boolean contains(int[] a, int from, int to, int x) {
        return Arrays.binarySearch(a, from, to, x) >= 0;
    }

    static boolean isSet(AtomicLongArray b, int v) {
        return (b.get(v >>> 6) & (1L << (v & 63))) != 0;
    }

    static void set(AtomicLongArray b, int v) {
        long mask = 1L << (v & 63);
        while (true) {
            long old = b.get(v >>> 6);
            if ((old & mask) != 0 || b.compareAndSet(v >>> 6, old, old | mask)) return;
        }
    }

    static int cardinality(AtomicLongArray b) {
        int c = 0;
        for (int i = 0; i < b.length(); i++) c += Long.bitCount(b.get(i));
        return c;
    }

    /** Per-thread work space: a stamped table of the accounts two transactions before v. */
    static final class Scratch {
        private static final ThreadLocal<Scratch> LOCAL = new ThreadLocal<>();
        final int[] stamp, y0, y1, y2, keys;
        final int[] count;
        int cur, nkeys;
        long ops;

        Scratch(int n) {
            stamp = new int[n];
            y0 = new int[n];
            y1 = new int[n];
            y2 = new int[n];
            count = new int[n];
            keys = new int[n];
        }

        static Scratch get(int n) {
            Scratch s = LOCAL.get();
            if (s == null || s.stamp.length != n) {
                s = new Scratch(n);
                LOCAL.set(s);
            }
            return s;
        }

        /** A cycle of m accounts through v in the graph (forward order), or null. */
        int[] search(CycleMembership g, int v, int m) {
            ops = 0;
            boolean forward = switch (g.orientation) {
                case "forward" -> true;
                case "reverse" -> false;
                default -> predictForward(g, v, m);
            };
            int[] oo = forward ? g.outOff : g.inOff, oa = forward ? g.outAdj : g.inAdj;
            int[] io = forward ? g.inOff : g.outOff, ia = forward ? g.inAdj : g.outAdj;
            int[] c = m == 3 ? three(v, oo, oa, io, ia) : m == 4 ? four(v, oo, oa, io, ia) : m == 5 ? five(v, oo, oa, io, ia)
                    : general(v, m, oo, oa, io, ia);
            g.examined.add(ops);
            if (c == null || forward) return c;
            int[] r = new int[c.length];                                     // a cycle of the reverse graph, read backwards
            r[0] = c[0];
            for (int i = 1; i < c.length; i++) r[i] = c[c.length - i];
            return r;
        }

        private static boolean predictForward(CycleMembership g, int v, int m) {
            long f2 = 0, b2 = 0, f1 = g.outOff[v + 1] - g.outOff[v], b1 = g.inOff[v + 1] - g.inOff[v];
            for (int i = g.outOff[v]; i < g.outOff[v + 1]; i++) f2 += g.outOff[g.outAdj[i] + 1] - g.outOff[g.outAdj[i]];
            for (int i = g.inOff[v]; i < g.inOff[v + 1]; i++) b2 += g.inOff[g.inAdj[i] + 1] - g.inOff[g.inAdj[i]];
            if (m == 3) return Math.min(f2, f1 * b1) <= Math.min(b2, b1 * f1);
            // the table costs the in-side two-step count once; the leaving paths repeat the out-side one
            return f2 <= b2;
        }

        /** v -> x -> w -> v. */
        private int[] three(int v, int[] oo, int[] oa, int[] io, int[] ia) {
            int iFrom = io[v], iTo = io[v + 1];
            for (int i = oo[v]; i < oo[v + 1]; i++) {
                int x = oa[i];
                int oFrom = oo[x], oTo = oo[x + 1];
                if (iTo - iFrom < oTo - oFrom) {
                    for (int j = iFrom; j < iTo; j++) {
                        ops++;
                        int w = ia[j];
                        if (w != x && contains(oa, oFrom, oTo, w)) return new int[] {v, x, w};
                    }
                } else {
                    for (int j = oFrom; j < oTo; j++) {
                        ops++;
                        int w = oa[j];
                        if (w != v && contains(ia, iFrom, iTo, w)) return new int[] {v, x, w};
                    }
                }
            }
            return null;
        }

        /** Tabulate z -> y -> v (z != v), keeping up to cap distinct y per z. */
        private void table(int v, int cap, int[] io, int[] ia) {
            cur++;
            nkeys = 0;
            for (int i = io[v]; i < io[v + 1]; i++) {
                int y = ia[i];
                for (int j = io[y]; j < io[y + 1]; j++) {
                    ops++;
                    int z = ia[j];
                    if (z == v) continue;
                    if (stamp[z] != cur) {
                        stamp[z] = cur;
                        count[z] = 0;
                        keys[nkeys++] = z;
                    }
                    int c = count[z];
                    if (c >= cap) continue;
                    if (c == 0) y0[z] = y;
                    else if (c == 1) y1[z] = y;
                    else y2[z] = y;
                    count[z] = c + 1;
                }
            }
        }

        /** A tabulated y for z that differs from a and b, or -1. */
        private int avoid(int z, int a, int b) {
            int c = count[z];
            if (c > 0 && y0[z] != a && y0[z] != b) return y0[z];
            if (c > 1 && y1[z] != a && y1[z] != b) return y1[z];
            if (c > 2 && y2[z] != a && y2[z] != b) return y2[z];
            return -1;
        }

        /** The last step of a leaving path ending at u: some z in out(u) that is tabulated, z not in {u, a, b}. */
        private int[] join(int u, int a, int b, int[] oo, int[] oa, int[] prefix) {
            int from = oo[u], to = oo[u + 1];
            if (nkeys < to - from) {
                for (int k = 0; k < nkeys; k++) {
                    ops++;
                    int z = keys[k];
                    if (z == a || z == b || z == u || !contains(oa, from, to, z)) continue;
                    int y = avoid(z, a, b == -1 ? u : b);
                    if (y >= 0) return close(prefix, z, y);
                }
            } else {
                for (int j = from; j < to; j++) {
                    ops++;
                    int z = oa[j];
                    if (stamp[z] != cur || z == a || z == b) continue;
                    int y = avoid(z, a, b == -1 ? u : b);
                    if (y >= 0) return close(prefix, z, y);
                }
            }
            return null;
        }

        private static int[] close(int[] prefix, int z, int y) {
            int[] c = Arrays.copyOf(prefix, prefix.length + 2);
            c[prefix.length] = z;
            c[prefix.length + 1] = y;
            return c;
        }

        /** v -> x -> z -> y -> v. */
        private int[] four(int v, int[] oo, int[] oa, int[] io, int[] ia) {
            table(v, 2, io, ia);
            if (nkeys == 0) return null;
            for (int i = oo[v]; i < oo[v + 1]; i++) {
                int x = oa[i];
                int[] c = join(x, x, -1, oo, oa, new int[] {v, x});
                if (c != null) return c;
            }
            return null;
        }

        /** v -> x1 -> x2 -> z -> y -> v. */
        private int[] five(int v, int[] oo, int[] oa, int[] io, int[] ia) {
            table(v, 3, io, ia);
            if (nkeys == 0) return null;
            for (int i = oo[v]; i < oo[v + 1]; i++) {
                int x1 = oa[i];
                for (int j = oo[x1]; j < oo[x1 + 1]; j++) {
                    ops++;
                    int x2 = oa[j];
                    if (x2 == v) continue;
                    int[] c = join(x2, x1, x2, oo, oa, new int[] {v, x1, x2});
                    if (c != null) return c;
                }
            }
            return null;
        }

        /** Plain depth-first search for lengths above 5 (not used by the experiments; kept for the tests). */
        private int[] general(int v, int m, int[] oo, int[] oa, int[] io, int[] ia) {
            int[] path = new int[m];
            path[0] = v;
            return dfs(path, 1, m, oo, oa, io, ia) ? path : null;
        }

        private boolean dfs(int[] path, int depth, int m, int[] oo, int[] oa, int[] io, int[] ia) {
            int u = path[depth - 1];
            for (int j = oo[u]; j < oo[u + 1]; j++) {
                ops++;
                int w = oa[j];
                boolean seen = false;
                for (int i = 0; i < depth; i++) if (path[i] == w) seen = true;
                if (seen) continue;
                path[depth] = w;
                if (depth == m - 1) {
                    if (contains(oa, oo[w], oo[w + 1], path[0])) return true;
                } else if (dfs(path, depth + 1, m, oo, oa, io, ia)) {
                    return true;
                }
            }
            return false;
        }
    }

    // ---------------------------------------------------------------- input and command line

    /** Reads v ID LABEL / e SOURCE TARGET LABEL AMOUNT; refuses more than one account or transaction label. */
    static CycleMembership load(Path path, String orientation) throws IOException {
        Map<Long, Integer> index = new HashMap<>();
        long[] ids = new long[1024];
        long[] pairs = new long[1 << 20];
        int np = 0, accounts = 0;
        long transactions = 0, selfDropped = 0;
        String vlabel = null, elabel = null;
        try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.isEmpty() || line.charAt(0) == '#') continue;
                String[] t = line.trim().split("\\s+");
                if (t[0].equals("v")) {
                    long id = Long.parseLong(t[1]);
                    if (index.putIfAbsent(id, accounts) != null) throw new IllegalArgumentException("account " + id + " declared twice");
                    if (accounts == ids.length) ids = Arrays.copyOf(ids, ids.length * 2);
                    ids[accounts++] = id;
                    if (vlabel == null) vlabel = t[2];
                    else if (!vlabel.equals(t[2])) throw new IllegalArgumentException("more than one account label; this program is for one label");
                } else if (t[0].equals("e")) {
                    transactions++;
                    Integer s = index.get(Long.parseLong(t[1])), d = index.get(Long.parseLong(t[2]));
                    if (s == null || d == null) throw new IllegalArgumentException("transaction with an undeclared account: " + line);
                    if (elabel == null) elabel = t[3];
                    else if (!elabel.equals(t[3])) throw new IllegalArgumentException("more than one transaction label; this program is for one label");
                    if (s.intValue() == d.intValue()) {
                        selfDropped++;
                        continue;
                    }
                    if (np == pairs.length) pairs = Arrays.copyOf(pairs, pairs.length * 2);
                    pairs[np++] = ((long) s << 32) | d;
                } else {
                    throw new IllegalArgumentException("unknown line: " + line);
                }
            }
        }
        long[] p = Arrays.copyOf(pairs, np);
        Arrays.parallelSort(p);
        int u = 0;
        for (int i = 0; i < p.length; i++) if (i == 0 || p[i] != p[i - 1]) p[u++] = p[i];
        System.out.printf("graph %s: %d accounts, %d transactions (%d self-transactions dropped), %d distinct ordered pairs%n",
                path.getFileName(), accounts, transactions, selfDropped, u);
        return new CycleMembership(Arrays.copyOf(ids, accounts), Arrays.copyOf(p, u), orientation);
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length || !args[i].startsWith("--")) throw new IllegalArgumentException("options come in pairs: --name value");
            opt.put(args[i], args[i + 1]);
        }
        for (String k : opt.keySet()) {
            if (!java.util.Set.of("--graph", "--min-length", "--max-length", "--min-support", "--threads", "--orientation", "--out").contains(k)) {
                throw new IllegalArgumentException("unknown option " + k);
            }
        }
        if (!opt.containsKey("--graph") || !opt.containsKey("--out")) throw new IllegalArgumentException("--graph and --out are required");
        int lo = Integer.parseInt(opt.getOrDefault("--min-length", "3")), hi = Integer.parseInt(opt.getOrDefault("--max-length", "5"));
        int tau = Integer.parseInt(opt.getOrDefault("--min-support", "1"));
        int threads = Integer.parseInt(opt.getOrDefault("--threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        String orientation = opt.getOrDefault("--orientation", "auto");
        if (!java.util.Set.of("auto", "forward", "reverse").contains(orientation)) throw new IllegalArgumentException("--orientation must be auto, forward or reverse");
        long t0 = System.nanoTime();
        CycleMembership g = load(Path.of(opt.get("--graph")), orientation);
        System.out.printf("strongly connected part: %d accounts, %d pairs%n", g.sccAccounts, g.sccEdges);
        System.out.printf("params: min_length=%d max_length=%d min_support=%d orientation=%s%n", lo, hi, tau, orientation);
        try (BufferedWriter w = Files.newBufferedWriter(Path.of(opt.get("--out")), StandardCharsets.UTF_8)) {
            w.write("kind\tlength\trole\taccount\n");
            for (int m = lo; m <= hi; m++) {
                long s0 = g.searches.sum(), f0 = g.found.sum(), e0 = g.examined.sum();
                AtomicLongArray c = g.members(m, threads);
                int size = cardinality(c);
                boolean reported = size >= tau;
                System.out.printf("length %d: %d accounts on a simple cycle (support %d, %s); searches=%d found=%d examined=%d%n", m, size,
                        size, reported ? "reported" : "below the threshold", g.searches.sum() - s0, g.found.sum() - f0, g.examined.sum() - e0);
                if (!reported) continue;
                for (String role : new String[] {"origin", "intermediate"}) {
                    for (int v = 0; v < g.n; v++) if (isSet(c, v)) w.write("closed\t" + m + "\t" + role + "\t" + g.originalId[v] + "\n");
                }
            }
        }
        System.out.printf("cycle membership elapsed ms: %d (machine-dependent)%n", (System.nanoTime() - t0) / 1_000_000);
    }
}
