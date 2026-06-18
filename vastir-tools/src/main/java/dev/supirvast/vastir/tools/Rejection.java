package dev.supirvast.vastir.tools;

/**
 * The witness for an unaccepted kernel — a clear, specific reason registration failed, paired with detail
 * (e.g. the {@code spirv-val} diagnostic, or the construct that could not be lowered). A front end renders
 * this against its own source; the orchestrator keeps the message concrete and unambiguous, never a bare
 * "unsupported".
 */
public record Rejection(String reason, String detail) implements Registration {

    public Rejection {
        if (reason == null || reason.isBlank()) {
            throw new IllegalArgumentException("a rejection must state a reason");
        }
        detail = detail == null ? "" : detail;
    }
}
