package dev.supirvast.vastir.pass;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;

import java.util.List;

/** One dispatch: a kernel, its bindings in order, the named buffers bound to them, and its invocations. */
public record Pass(String name, Function kernel, List<Buffer> bindings, List<String> buffers, int invocations) {
    public Pass {
        if (bindings.size() != buffers.size()) {
            throw new IllegalArgumentException(name + ": " + bindings.size() + " bindings but " + buffers.size()
                    + " buffers");
        }
    }
}
