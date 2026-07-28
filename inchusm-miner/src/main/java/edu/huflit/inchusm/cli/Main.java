package edu.huflit.inchusm.cli;

import edu.huflit.inchusm.csp.BallCache;
import edu.huflit.inchusm.csp.CMNITable;
import edu.huflit.inchusm.csp.CSPSolver;
import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.graph.GraphLoader;
import edu.huflit.inchusm.mining.BHUS;
import edu.huflit.inchusm.mining.BHUSInc;
import edu.huflit.inchusm.mining.HUSM;
import edu.huflit.inchusm.mining.IncHUSM;
import edu.huflit.inchusm.mining.Measure;
import edu.huflit.inchusm.mining.PatternState;
import edu.huflit.inchusm.mining.RelativeThreshold;
import edu.huflit.inchusm.mining.UtilityTable;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * CLI with three subcommands.
 *
 * <pre>
 * mine    -  static HUSM on one graph:
 *   mine --graph data.lg (--util u.tsv | --util-gauss 3.0,1.0,42) --minutil 50
 *        [--mer 1] [--bound ceu|eu|ivu] [--max-edges 0] [--out hus.jsonl] [--bhus]
 *
 * stream  -  incremental IncHUSM over a sequence of disjoint snapshots:
 *   stream (--snapshot-dir DIR | --snapshots a.lg,b.lg,...)
 *          (--util u.tsv | --util-gauss 3.0,1.0,42) --minvalue 50
 *          [--measure utility|support] [--mer 1] [--bound ceu] [--max-edges 5]
 *          [--compare-static] [--out per_snapshot.csv]
 *
 * match   -  locate a GIVEN pattern set in a graph and report which vertices realize it:
 *   match --graph data.lg (--util u.tsv | --util-gauss ...) --patterns codes.txt
 *         [--mer 1] --out images.csv
 *   One canonical DFS code per line in codes.txt (the key() form written by `mine`).
 *   Output rows are pattern,role,vertex  -  the image sets of each pattern. This turns a
 *   mined pattern set into per-vertex evidence, which is what an actor-level study needs;
 *   mining a vocabulary on a training window and matching it elsewhere keeps the pattern
 *   set frozen, exactly as the utility buckets are.
 * </pre>
 *
 * {@code --window W} keeps only the last W snapshots (sliding window); the default 0 is
 * the add-only mode. Under a window {@code --compare-static} re-runs static HUSM on the
 * WINDOW union rather than the accumulated union, since the window is what is maintained.
 *
 * Snapshots are read in filename order (a .lg per time step). {@code --compare-static}
 * re-runs static HUSM on the accumulated union after each snapshot and checks the
 * high set matches  -  this yields the correctness confirmation and the speedup
 * number (incremental vs static re-run) that experiment E1/E2 report.
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            usage();
            System.exit(2);
        }
        Map<String, String> opt = parse(args);
        switch (args[0]) {
            case "mine" -> mine(opt);
            case "stream" -> stream(opt);
            case "match" -> match(opt);
            default -> {
                usage();
                System.exit(2);
            }
        }
    }

    // ---- static mining ----------------------------------------------------

    private static void mine(Map<String, String> opt) throws Exception {
        Path graphPath = Path.of(require(opt, "--graph"));
        double minUtil = Double.parseDouble(require(opt, "--minutil"));
        int mer = Integer.parseInt(opt.getOrDefault("--mer", "1"));
        HUSM.Bound bound = HUSM.Bound.valueOf(opt.getOrDefault("--bound", "ceu").toUpperCase(Locale.ROOT));
        int maxEdges = Integer.parseInt(opt.getOrDefault("--max-edges", "0"));
        Measure measure = Measure.valueOf(opt.getOrDefault("--measure", "utility").toUpperCase(Locale.ROOT));

        Graph g = GraphLoader.loadLG(graphPath);
        // FLOW scores by money on the edges, not by label utility, so the table is optional there.
        Map<Integer, Double> table = measure == Measure.FLOW
                ? Map.of()
                : utilities(opt, labelsOf(List.of(g)));

        System.out.printf(Locale.ROOT, "graph: %s (%d vertices, %d edges, %d labels)%n",
                graphPath.getFileName(), g.n(), g.m(), table.size());
        System.out.printf(Locale.ROOT, "config: minUtil=%s MER=%d measure=%s bound=%s maxEdges=%s%n",
                minUtil, mer, measure, bound, maxEdges > 0 ? maxEdges : "unlimited");

        HUSM.Result res = HUSM.mine(g, new UtilityTable(table),
                new HUSM.Config(minUtil, mer, bound, maxEdges, false, measure));

        BHUS.Sets bhus = BHUS.extract(res.hus.values());
        System.out.printf(Locale.ROOT,
                "evaluated=%d candidates=%d HUS=%d BHUS=%d runtime=%dms%n",
                res.stats.evaluated, res.stats.candidates, res.hus.size(),
                bhus.all().size(), res.stats.runtimeMs);

        if (opt.containsKey("--out")) {
            Path out = Path.of(opt.get("--out"));
            if (out.getParent() != null) Files.createDirectories(out.getParent());
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
                boolean markBhus = opt.containsKey("--bhus");
                for (HUSM.PatternResult p : res.hus.values()) {
                    String key = p.code().key();
                    StringBuilder sb = new StringBuilder();
                    sb.append("{\"code\":\"").append(key).append('"');
                    sb.append(",\"vertices\":").append(p.code().nVertices());
                    sb.append(",\"edges\":").append(p.code().size());
                    sb.append(",\"cmni\":").append(p.eval().cmni());
                    if (measure == Measure.FLOW) {
                        sb.append(String.format(Locale.ROOT, ",\"bottleneck\":%.6f", p.eval().bottleneck()));
                        sb.append(String.format(Locale.ROOT, ",\"flow\":%.6f", p.eval().flow()));
                        sb.append(String.format(Locale.ROOT, ",\"utility\":%.6f", p.eval().flow()));
                    } else {
                        sb.append(String.format(Locale.ROOT, ",\"utility\":%.6f", p.eval().utility()));
                        sb.append(String.format(Locale.ROOT, ",\"ceu\":%.6f", p.eval().ceu()));
                    }
                    if (markBhus) {
                        sb.append(",\"bhus_max\":").append(bhus.maxBoundary().contains(key));
                        sb.append(",\"bhus_min\":").append(bhus.minBoundary().contains(key));
                    }
                    sb.append('}');
                    w.println(sb);
                }
            }
            System.out.println("wrote " + res.hus.size() + " patterns to " + out);
        }
    }

    // ---- matching a given pattern set --------------------------------------

    /**
     * Locate each supplied pattern in the graph and write its image sets.
     *
     * The miner reports which patterns are high, not which vertices realize them, because
     * mining only ever needs the counts (Lemma L1). Recovering the vertices is a plain
     * re-run of the same constraint solver per pattern, which is what this does.
     */
    private static void match(Map<String, String> opt) throws Exception {
        Path graphPath = Path.of(require(opt, "--graph"));
        Path patternPath = Path.of(require(opt, "--patterns"));
        int mer = Integer.parseInt(opt.getOrDefault("--mer", "1"));

        Graph g = GraphLoader.loadLG(graphPath);
        Map<Integer, Double> table = utilities(opt, labelsOf(List.of(g)));
        UtilityTable ext = new UtilityTable(table);
        BallCache balls = new BallCache(g, mer);

        List<DFSCode> codes = new ArrayList<>();
        for (String line : Files.readAllLines(patternPath, StandardCharsets.UTF_8)) {
            String s = line.trim();
            if (!s.isEmpty()) codes.add(DFSCode.parse(s));
        }
        System.out.printf(Locale.ROOT, "match: %s (%d vertices, %d edges) against %d patterns%n",
                graphPath.getFileName(), g.n(), g.m(), codes.size());

        Path out = Path.of(require(opt, "--out"));
        if (out.getParent() != null) Files.createDirectories(out.getParent());
        long t0 = System.nanoTime();
        int matched = 0;
        long rows = 0;
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            w.println("pattern,role,vertex");
            for (DFSCode code : codes) {
                CMNITable images = new CSPSolver(g, code, balls).solve();
                boolean any = false;
                // a DFS code contains commas, so it must be quoted to stay one CSV field
                String key = '"' + code.key() + '"';
                for (int v = 0; v < images.nColumns(); v++) {
                    for (int x : images.images(v)) {
                        w.printf(Locale.ROOT, "%s,%d,%d%n", key, v, x);
                        rows++;
                        any = true;
                    }
                }
                if (any) matched++;
            }
        }
        System.out.printf(Locale.ROOT, "matched %d/%d patterns, %d image rows (%dms) -> %s%n",
                matched, codes.size(), rows, (System.nanoTime() - t0) / 1_000_000, out);
    }

    // ---- incremental streaming -------------------------------------------

    private static void stream(Map<String, String> opt) throws Exception {
        List<Path> paths = snapshotPaths(opt);
        // IncHUSM-mu: relative threshold with a pre-large band (mu-lower defaults to mu-upper).
        RelativeThreshold rel = opt.containsKey("--mu-upper")
                ? new RelativeThreshold(Double.parseDouble(require(opt, "--mu-upper")),
                        Double.parseDouble(opt.getOrDefault("--mu-lower", opt.get("--mu-upper"))))
                : null;
        double minValue = (rel != null && !opt.containsKey("--minvalue"))
                ? 0.0 : Double.parseDouble(require(opt, "--minvalue"));
        int mer = Integer.parseInt(opt.getOrDefault("--mer", "1"));
        HUSM.Bound bound = HUSM.Bound.valueOf(opt.getOrDefault("--bound", "ceu").toUpperCase(Locale.ROOT));
        Measure measure = Measure.valueOf(opt.getOrDefault("--measure", "utility").toUpperCase(Locale.ROOT));
        int maxEdges = Integer.parseInt(opt.getOrDefault("--max-edges", "5"));
        boolean compare = opt.containsKey("--compare-static");
        // sliding window: keep only the last W snapshots (0 = add-only, the original mode)
        int window = Integer.parseInt(opt.getOrDefault("--window", "0"));
        BHUSInc bhus = opt.containsKey("--bhus-inc") ? new BHUSInc() : null;

        List<Graph> snaps = new ArrayList<>();
        for (Path p : paths) snaps.add(GraphLoader.loadLG(p));
        UtilityTable ext = new UtilityTable(utilities(opt, labelsOf(snaps)));

        System.out.printf(Locale.ROOT,
                "stream: %d snapshots | measure=%s minValue=%s MER=%d bound=%s maxEdges=%d window=%s%n",
                snaps.size(), measure, minValue, mer, bound, maxEdges,
                window > 0 ? String.valueOf(window) : "add-only");

        IncHUSM inc = new IncHUSM(
                new IncHUSM.Config(minValue, mer, bound, measure, maxEdges), ext, rel, window);

        PrintWriter csv = null;
        if (opt.containsKey("--out")) {
            Path out = Path.of(opt.get("--out"));
            if (out.getParent() != null) Files.createDirectories(out.getParent());
            csv = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8));
            csv.println("snapshot,vertices,edges,tracked,high_total,newly_high,seg_last,"
                    + "inc_ms,static_ms,static_high,match,max_bhus,min_bhus,scale,large");
        }

        long incCumMs = 0;
        long staticCumMs = 0;
        List<Graph> seen = new ArrayList<>();
        for (int i = 0; i < snaps.size(); i++) {
            Graph gt = snaps.get(i);
            List<PatternState> newly = inc.update(gt);
            if (bhus != null) bhus.absorb(newly.stream().map(PatternState::code).toList());
            long muLarge = -1;   // IncHUSM-mu: patterns at or above the upper (large) threshold now
            if (rel != null) {
                double thetaUpper = rel.upper(inc.scale());
                muLarge = 0;
                for (PatternState ps : inc.highPatterns()) {
                    if (measure.value(ps.evaluation()) >= thetaUpper) muLarge++;
                }
            }
            IncHUSM.Stats st = inc.stats();
            incCumMs += st.lastUpdateMs;
            seen.add(gt);

            long staticMs = -1;
            int staticHigh = -1;
            String match = "";
            if (compare) {
                // add-only compares against the accumulated union; a sliding window must
                // compare against the WINDOW union, which is what it claims to maintain.
                List<Graph> ref = window > 0
                        ? seen.subList(Math.max(0, seen.size() - window), seen.size())
                        : seen;
                Graph union = disjointUnion(ref);
                long s0 = System.nanoTime();
                Set<String> stat = staticHighSet(union, ext, minValue, mer, bound, measure, maxEdges);
                staticMs = (System.nanoTime() - s0) / 1_000_000;
                staticCumMs += staticMs;
                staticHigh = stat.size();
                Set<String> got = new TreeSet<>();
                for (PatternState ps : inc.highPatterns()) got.add(ps.code().key());
                match = got.equals(stat) ? "OK" : "MISMATCH";
                if (!match.equals("OK")) {
                    // Report the exact difference: a pattern missing while already in the
                    // store points at its maintained value, one absent from the store points
                    // at the traversal instead.
                    Set<String> missing = new TreeSet<>(stat);
                    missing.removeAll(got);
                    Set<String> extra = new TreeSet<>(got);
                    extra.removeAll(stat);
                    for (String k : missing) {
                        PatternState ps = inc.store().get(k);
                        System.out.printf(Locale.ROOT, "    MISSING %s | inStore=%s%s%n", k,
                                ps != null,
                                ps == null ? "" : String.format(Locale.ROOT,
                                        " value=%.2f", measure.value(ps.evaluation())));
                    }
                    for (String k : extra) System.out.printf(Locale.ROOT, "    EXTRA   %s%n", k);
                }
            }

            System.out.printf(Locale.ROOT,
                    "t=%2d n=%-6d e=%-6d tracked=%-6d high=%-5d (+%d) seg=%-6d inc=%dms%s%n",
                    i + 1, gt.n(), gt.m(), st.patternsTracked, st.highTotal, newly.size(),
                    st.segmentsComputedLast, st.lastUpdateMs,
                    compare ? String.format(Locale.ROOT, " static=%dms [%s]", staticMs, match) : "");

            if (csv != null) {
                csv.printf(Locale.ROOT, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%.1f,%d%n",
                        i + 1, gt.n(), gt.m(), st.patternsTracked, st.highTotal, newly.size(),
                        st.segmentsComputedLast, st.lastUpdateMs, staticMs, staticHigh, match,
                        bhus != null ? bhus.maxSize() : -1, bhus != null ? bhus.minSize() : -1,
                        inc.scale(), muLarge);
                csv.flush(); // long run: write per-snapshot progress so a mid-run kill loses no data
            }
        }
        if (csv != null) csv.close();

        System.out.printf(Locale.ROOT, "TOTAL incremental=%dms | high patterns=%d%n",
                incCumMs, inc.highPatterns().size());
        if (compare && incCumMs > 0) {
            System.out.printf(Locale.ROOT, "TOTAL static-rerun=%dms | speedup=%.2fx%n",
                    staticCumMs, staticCumMs / (double) Math.max(1, incCumMs));
        }
    }

    /** Static reference high set on a graph, for either measure. */
    private static Set<String> staticHighSet(Graph g, UtilityTable ext, double minValue,
                                             int mer, HUSM.Bound bound, Measure measure, int maxEdges) {
        Set<String> out = new TreeSet<>();
        if (measure == Measure.UTILITY) {
            HUSM.Result r = HUSM.mine(g, ext, new HUSM.Config(minValue, mer, bound, maxEdges));
            out.addAll(r.hus.keySet());
        } else {
            // support mode reference: enumerate every pattern up to maxEdges
            // (threshold 0) and keep those with CMNI >= minValue. Only used with
            // --compare-static on small/medium unions for validation.
            HUSM.Result r = HUSM.mine(g, ext, new HUSM.Config(0.0, mer, bound, maxEdges));
            for (HUSM.PatternResult p : r.hus.values()) {
                if (p.eval().cmni() >= minValue) out.add(p.code().key());
            }
        }
        return out;
    }

    private static Graph disjointUnion(List<Graph> parts) {
        int total = 0;
        for (Graph p : parts) total += p.n();
        int[] vlabel = new int[total];
        List<int[]> edges = new ArrayList<>();
        int offset = 0;
        for (Graph p : parts) {
            for (int u = 0; u < p.n(); u++) vlabel[offset + u] = p.vlabel(u);
            for (int u = 0; u < p.n(); u++) {
                int[] nbrs = p.neighbors(u);
                int[] labs = p.neighborEdgeLabels(u);
                for (int k = 0; k < nbrs.length; k++) {
                    if (u < nbrs[k]) edges.add(new int[] {offset + u, offset + nbrs[k], labs[k]});
                }
            }
            offset += p.n();
        }
        return new Graph(vlabel, edges);
    }

    // ---- shared helpers ---------------------------------------------------

    private static List<Path> snapshotPaths(Map<String, String> opt) throws Exception {
        if (opt.containsKey("--snapshots")) {
            List<Path> out = new ArrayList<>();
            for (String s : opt.get("--snapshots").split(",")) out.add(Path.of(s.trim()));
            return out;
        }
        Path dir = Path.of(require(opt, "--snapshot-dir"));
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> p.toString().endsWith(".lg"))
                    .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                    .toList();
        }
    }

    private static Set<Integer> labelsOf(List<Graph> graphs) {
        Set<Integer> labels = new HashSet<>();
        for (Graph g : graphs) {
            for (int v = 0; v < g.n(); v++) labels.add(g.vlabel(v));
        }
        return labels;
    }

    private static Map<Integer, Double> utilities(Map<String, String> opt, Set<Integer> labels)
            throws Exception {
        if (opt.containsKey("--util")) {
            return GraphLoader.loadUtilityTable(Path.of(opt.get("--util")));
        }
        if (opt.containsKey("--util-gauss")) {
            String[] p = opt.get("--util-gauss").split(",");
            return GraphLoader.gaussianUtilities(labels,
                    Double.parseDouble(p[0]), Double.parseDouble(p[1]), Long.parseLong(p[2]));
        }
        throw new IllegalArgumentException("need --util FILE or --util-gauss MU,SIGMA,SEED");
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> opt = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    opt.put(args[i], args[++i]);
                } else {
                    opt.put(args[i], "true"); // boolean flag
                }
            }
        }
        return opt;
    }

    private static String require(Map<String, String> opt, String key) {
        String v = opt.get(key);
        if (v == null) throw new IllegalArgumentException("missing required option " + key);
        return v;
    }

    private static void usage() {
        System.err.println("""
                usage:
                  mine   --graph FILE.lg --minutil X (--util F | --util-gauss MU,SIGMA,SEED)
                         [--mer 1] [--bound ceu|eu|ivu] [--max-edges 0] [--out hus.jsonl] [--bhus]
                  stream (--snapshot-dir DIR | --snapshots a.lg,b.lg,...) --minvalue X
                         (--util F | --util-gauss MU,SIGMA,SEED) [--measure utility|support]
                         [--mer 1] [--bound ceu] [--max-edges 5] [--compare-static] [--out csv]
                         [--bhus-inc]                          (maintain BHUS boundaries per snapshot)
                         [--window W]                          (sliding window; 0 = add-only)
                         [--mu-upper FRAC [--mu-lower FRAC]]    (IncHUSM-mu; --minvalue optional)
                  match  --graph FILE.lg --patterns codes.txt --out images.csv
                         (--util F | --util-gauss MU,SIGMA,SEED) [--mer 1]
                         (one canonical DFS code per line; writes pattern,role,vertex)""");
    }
}
