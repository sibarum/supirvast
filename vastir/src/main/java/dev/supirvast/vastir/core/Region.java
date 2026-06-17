package dev.supirvast.vastir.core;

import java.util.List;

/**
 * A structured region: an ordered sequence of statements forming a single-entry, single-exit body.
 *
 * <p>This is the shared structural notion the naming convention calls out — the same region concept both the
 * SPIR-V structured CFG and the future Truffle tree build on. Nested regions (inside if/loop statements) give
 * structured control flow without a flat block graph.
 */
public record Region(List<Statement> statements) {

    public Region {
        statements = List.copyOf(statements);
    }

    public static Region of(Statement... statements) {
        return new Region(List.of(statements));
    }
}
