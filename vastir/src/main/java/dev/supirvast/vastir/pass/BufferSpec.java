package dev.supirvast.vastir.pass;

import dev.supirvast.vastir.type.Type;

/**
 * A buffer every pass refers to by name: its element type and length.
 *
 * @param readout whether the host reads it after every run, a few words at a time: kept in memory the host can see,
 *                so it is read where it is rather than copied out by a submission of its own
 */
public record BufferSpec(String name, Type element, int length, boolean readout) {

    public BufferSpec(String name, Type element, int length) {
        this(name, element, length, false);
    }

    /** A buffer the host reads after runs: see {@link #readout()}. */
    public static BufferSpec readout(String name, Type element, int length) {
        return new BufferSpec(name, element, length, true);
    }
}
