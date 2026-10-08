package dev.supirvast.vastir.pass;

import java.util.Map;

/**
 * A multi-pass program that names its buffers: what a runner allocates before it runs the program's {@link Pass
 * passes} over them.
 *
 * <p>A program is data: which kernel, over which named buffers, with how many invocations, in order. Nothing that
 * describes one runs anything, so a runner on any backend takes the same description, and the tests and the
 * application run one program rather than two copies of its wiring. {@code PassRunner} in vastir-tools is that
 * runner, on the GPU or the CPU.
 */
public interface Buffered {

    /** Every buffer the passes name, in a stable order. */
    Map<String, BufferSpec> buffers();
}
