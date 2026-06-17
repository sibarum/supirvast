package dev.supirvast.vastir.core;

import java.util.ArrayList;
import java.util.List;

/**
 * The top of the core IR: the functions and entry points of a shader module.
 *
 * <p>Target-neutral — it carries no SPIR-V ids, blocks, or capabilities. {@code CoreToSpirv} lowers it to a
 * {@link dev.supirvast.vastir.binary.SpirvModule}; a future {@code CoreToTruffle} consumes the same shape.
 */
public final class CoreModule {

    private final List<Function> functions = new ArrayList<>();
    private final List<EntryPoint> entryPoints = new ArrayList<>();

    public CoreModule addFunction(Function function) {
        functions.add(function);
        return this;
    }

    public CoreModule addEntryPoint(EntryPoint entryPoint) {
        if (!functions.contains(entryPoint.function())) {
            functions.add(entryPoint.function());
        }
        entryPoints.add(entryPoint);
        return this;
    }

    public List<Function> functions() {
        return List.copyOf(functions);
    }

    public List<EntryPoint> entryPoints() {
        return List.copyOf(entryPoints);
    }
}
