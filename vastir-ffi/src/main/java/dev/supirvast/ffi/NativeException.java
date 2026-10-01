package dev.supirvast.ffi;

/**
 * Thrown when a native call fails — a missing symbol, a null/failure return code, or a {@code Throwable} escaping
 * an {@code invokeExact}. Every binding wrapper funnels failures here so native errors are never swallowed and
 * always carry the offending call's name. See {@code docs/native-bindings.md} §4.
 *
 * <p>Not final: a native failure whose <em>recovery</em> differs in kind deserves to be told apart by type
 * rather than by parsing a message. Device loss is the case that forced it — everything built from the device
 * is gone, so a caller's response is to rebuild rather than to retry, and a subscriber several layers up needs
 * to know which of those it is looking at. Subclass only for that reason; a subclass per failing call would be
 * exactly the taxonomy this class exists to avoid.
 */
public class NativeException extends RuntimeException {

    public NativeException(String message) {
        super(message);
    }

    public NativeException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Wrap a {@code Throwable} thrown out of an {@code invokeExact} for {@code call}. An existing
     * {@link NativeException} passes through unchanged so the original message survives.
     */
    public static NativeException rethrow(String call, Throwable cause) {
        if (cause instanceof NativeException ne) {
            return ne;
        }
        return new NativeException("native call failed: " + call, cause);
    }
}
