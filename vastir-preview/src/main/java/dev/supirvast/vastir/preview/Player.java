package dev.supirvast.vastir.preview;

/**
 * A cooperative worker in the {@link Conductor}'s orchestra — ticked on its cadence, retiring when it returns
 * {@code false}. In the full Orchestration API a Player is a Pontif conduit; this is the host-level interface
 * the Conductor spike drives.
 */
@FunctionalInterface
public interface Player {

    /** Play one tick at logical time {@code nowNanos}; return {@code false} to retire from the orchestra. */
    boolean play(long nowNanos);
}
