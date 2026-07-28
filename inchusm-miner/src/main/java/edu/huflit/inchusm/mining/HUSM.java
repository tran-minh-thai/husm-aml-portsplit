package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.csp.BallCache;
import edu.huflit.inchusm.dfs.DFSCode;
import edu.huflit.inchusm.dfs.EdgeTuple;
import edu.huflit.inchusm.dfs.MinCode;
import edu.huflit.inchusm.graph.Graph;
import edu.huflit.inchusm.mining.ValueCompute.Evaluation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * HUSM Algorithms 1-2 (Chen et al. 2024). Two deliberately separate thresholds
 * (utility is NOT downward-closed, the bound is):
 *
 *   u(P)     >= minUtil  ->  P is reported as a high-utility subgraph;
 *   bound(P) >= minUtil  ->  P is a CANDIDATE and gets extended further.
 *
 * A pattern failing the first test but passing the second is still extended  - 
 * getting this wrong silently loses results. Candidate generation is label-driven
 * over the retained 1-edge set (Alg. 1 line 2: types whose bound >= minUtil;
 * safe by downward closure of the bound), with gSpan rightmost-path extensions
 * and the {@link MinCode#isMin} anti-duplicate prune.
 *
 * {@code maxEdges > 0} caps pattern size  -  used both for test equivalence with
 * brute force and as the AML "&lt;= 5-node motif" setting of the project.
 */
public final class HUSM {

    public enum Bound { IVU, EU, CEU }

    public record Config(double minUtil, int mer, Bound bound, int maxEdges,
                         boolean recordEvaluations, Measure measure) {

        public Config(double minUtil, int mer, Bound bound, int maxEdges, boolean recordEvaluations) {
            this(minUtil, mer, bound, maxEdges, recordEvaluations, Measure.UTILITY);
        }

        public Config(double minUtil, int mer, Bound bound, int maxEdges) {
            this(minUtil, mer, bound, maxEdges, false, Measure.UTILITY);
        }
    }

    public record PatternResult(DFSCode code, Evaluation eval) {
    }

    public static final class Stats {
        public long evaluated;
        public long candidates;
        public long runtimeMs;
    }

    public static final class Result {
        public final Map<String, PatternResult> hus = new LinkedHashMap<>();
        public final Stats stats = new Stats();
        public final List<Evaluation> allEvaluations = new ArrayList<>();
    }

    private final Graph g;
    private final UtilityTable ext;
    private final Config cfg;
    private final BallCache balls;
    private final Result result = new Result();
    private Alphabet alphabet;

    private HUSM(Graph g, UtilityTable ext, Config cfg) {
        this.g = g;
        this.ext = ext;
        this.cfg = cfg;
        this.balls = new BallCache(g, cfg.mer());
    }

    public static Result mine(Graph g, UtilityTable ext, Config cfg) {
        return new HUSM(g, ext, cfg).run();
    }

    private Result run() {
        long t0 = System.nanoTime();

        // Alg. 1 line 2: retain 1-edge types whose selected bound reaches minUtil
        List<Graph.EdgeType> retained = new ArrayList<>();
        List<DFSCode> startCodes = new ArrayList<>();
        List<Evaluation> startEvals = new ArrayList<>();
        for (Graph.EdgeType type : g.distinctEdgeTypes()) {
            DFSCode c1 = DFSCode.single(new EdgeTuple(0, 1, type.la(), type.le(), type.lb()));
            Evaluation ev = evaluate(c1);
            if (cfg.measure().bound(ev, cfg.bound()) >= cfg.minUtil()) {
                result.stats.candidates++;
                retained.add(type);
                startCodes.add(c1);
                startEvals.add(ev);
            }
        }
        alphabet = new Alphabet(retained);

        // Alg. 1 lines 3-8
        for (int i = 0; i < startCodes.size(); i++) {
            DFSCode c1 = startCodes.get(i);
            Evaluation ev = startEvals.get(i);
            if (cfg.measure().value(ev) >= cfg.minUtil()) addResult(c1, ev);
            extend(c1);
        }

        result.stats.runtimeMs = (System.nanoTime() - t0) / 1_000_000;
        return result;
    }

    /** Alg. 2 (SubgraphExtension): generate children, evaluate, report, recurse. */
    private void extend(DFSCode code) {
        if (cfg.maxEdges() > 0 && code.size() >= cfg.maxEdges()) return;

        for (EdgeTuple e : Extensions.rightmost(code, alphabet)) {
            DFSCode child = code.extend(e);
            if (!MinCode.isMin(child)) continue; // anti-duplicate prune
            Evaluation ev = evaluate(child);
            if (cfg.measure().value(ev) >= cfg.minUtil()) addResult(child, ev);
            if (cfg.measure().bound(ev, cfg.bound()) >= cfg.minUtil()) {
                result.stats.candidates++;
                extend(child);
            }
        }
    }

    private Evaluation evaluate(DFSCode code) {
        Evaluation ev = cfg.measure() == Measure.FLOW
                ? ValueCompute.evaluateFlow(g, code, balls)
                : ValueCompute.evaluate(g, code, ext, balls);
        result.stats.evaluated++;
        if (cfg.recordEvaluations()) result.allEvaluations.add(ev);
        return ev;
    }

    private void addResult(DFSCode code, Evaluation ev) {
        result.hus.put(code.key(), new PatternResult(code, ev));
    }
}
