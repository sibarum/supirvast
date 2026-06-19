package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * A push-constant block — a small, fast-to-update uniform record the host sets per draw (e.g. a model-view-
 * projection matrix). Declared as an ordered list of typed {@link Member}s and read with
 * {@link Expr.PushConstantRead}. A module uses at most one push-constant block (the SPIR-V rule), lowered to a
 * {@code PushConstant} {@code Block}-decorated struct.
 */
public record PushConstants(List<Member> members) {

    public PushConstants {
        members = List.copyOf(members);
    }

    public record Member(String name, Type type) {}

    /** A single-member block (the common case, e.g. just the MVP matrix). */
    public static PushConstants of(String name, Type type) {
        return new PushConstants(List.of(new Member(name, type)));
    }

    /** Reads member {@code index} of this block. */
    public Expr.PushConstantRead read(int index) {
        return new Expr.PushConstantRead(this, index);
    }
}
