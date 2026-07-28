package edu.huflit.inchusm.mining;

/**
 * Relative (data-scaled) utility threshold with a pre-large safety band, for IncHUSM-mu.
 *
 * The absolute threshold minValue in {@link IncHUSM} is fixed. When the threshold is instead
 * a fraction mu of the data size, the effective absolute threshold grows as the stream
 * accumulates, which breaks the "once high, always high" monotonicity that exact IncHUSM
 * relies on: a pattern high at snapshot t can fall below the (higher) threshold at t+1.
 *
 * Following the pre-large idea from incremental high-utility itemset mining, two fractions
 * bracket a safety band:
 *   muUpper  a pattern is LARGE (reported) when utility >= muUpper * scale;
 *   muLower  a pattern is PRE-LARGE (tracked as a buffer, not reported) when
 *            muLower * scale <= utility < muUpper * scale.
 * A pattern below muLower * scale is SMALL and not tracked.
 *
 * Guarantee. Only patterns at or above the upper threshold are reported, so there is NO
 * FALSE POSITIVE. A small pattern cannot cross the upper threshold within updates whose
 * accumulated added scale stays under {@link #safetyBound}, because it would first appear
 * in the pre-large band; so nothing large is missed while the bound holds. Beyond the bound
 * a full re-mine is required to restore exactness.
 *
 * IMPORTANT  -  the bound is NOT enforced here. This engine tracks down to muLower * scale and
 * never re-mines; {@link #safetyBound} states the condition, it does not police it. Whether
 * the bound actually holds on a given stream is an empirical question the driver answers, and
 * on the Elliptic++ actor stream the answer is that it does not: a snapshot there adds a
 * median 3.2% of the accumulated scale (46% at the head of the stream) while the bound at
 * mu=(0.02, 0.01) is 1.02% of it, so an exact run would have to re-mine at 46 of 49
 * snapshots. See e5c_inchusm_mu.py, which reports that schedule. The pre-large technique
 * assumes increments that are small relative to the accumulated data; a 49-snapshot stream
 * does not provide them.
 *
 * "scale" is the accumulated external-utility weight of the data seen so far (the
 * utility analogue of the transaction count used for relative support).
 */
public record RelativeThreshold(double muUpper, double muLower) {

    public RelativeThreshold {
        if (!(0 < muLower && muLower <= muUpper && muUpper <= 1)) {
            throw new IllegalArgumentException("require 0 < muLower <= muUpper <= 1");
        }
    }

    /** Absolute upper (large) threshold at the current data scale; patterns >= this are reported. */
    public double upper(double scale) {
        return muUpper * scale;
    }

    /** Absolute lower (pre-large) threshold; IncHUSM tracks patterns down to this value. */
    public double lower(double scale) {
        return muLower * scale;
    }

    /**
     * Largest added scale that may accumulate since the last full re-mine while the answer
     * stays exact: as long as the accumulated added scale is at or below this, a small
     * pattern cannot reach the upper threshold without first being seen in the pre-large
     * band. Adapted from the pre-large itemset safety bound (Hong et al.).
     */
    public double safetyBound(double scale) {
        return (muUpper - muLower) * scale / (1.0 - muUpper);
    }
}
