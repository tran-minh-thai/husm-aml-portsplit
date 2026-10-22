package edu.huflit.inchusm.mining;

import edu.huflit.inchusm.mining.ValueCompute.Evaluation;

/**
 * What IncHUSM maintains (the flag exists from the first
 * version so support-only is a genuine ablation, not a bolt-on).
 *
 * UTILITY  -  value = u(P) (Def. 8), pruning bound = the chosen upper bound
 *           (IVU/EU/CEU). Utility is not anti-monotone, hence the separate bound.
 * SUPPORT  -  value = CMNI(P). CMNI IS anti-monotone (HUSM paper, Lemma 2), so it is its own
 *           bound; this reduces IncHUSM to incremental frequent-subgraph mining
 *           (DyFSM-style) within the very same framework  -  the head-to-head that
 *           answers "does utility earn its keep?" (the support-vs-utility ablation).
 * FLOW     -  value = CMNI(P) x B(P) = u_flow (the bottleneck measure). Anti-monotone (see
 *           {@link edu.huflit.inchusm.csp.BottleneckSolver}), so like SUPPORT it is
 *           its own bound; no CEU/EU/IVU needed because the money measure, unlike
 *           HUSM's mean-label utility, IS downward-closed.
 */
public enum Measure {
    UTILITY,
    SUPPORT,
    FLOW;

    public double value(Evaluation ev) {
        return switch (this) {
            case UTILITY -> ev.utility();
            case SUPPORT -> ev.cmni();
            case FLOW -> ev.flow();
        };
    }

    public double bound(Evaluation ev, HUSM.Bound b) {
        return switch (this) {
            case UTILITY -> ev.bound(b);
            case SUPPORT -> ev.cmni();
            case FLOW -> ev.flow();
        };
    }
}
