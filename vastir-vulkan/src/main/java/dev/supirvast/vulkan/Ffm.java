package dev.supirvast.vulkan;

import dev.supirvast.ffi.NativeException;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Small shared helpers for filling and reading Vulkan structs by field name, and invoking resolved command
 * handles. Field access derives every offset from the layout (never a literal), per the binding convention
 * ({@code docs/native-bindings.md} §4.3). Used by the Vulkan runtime classes to keep struct-heavy code readable.
 */
public final class Ffm {

    private Ffm() {
    }

    public static long off(GroupLayout layout, String field) {
        return layout.byteOffset(PathElement.groupElement(field));
    }

    public static void si(MemorySegment s, GroupLayout l, String f, int v) {
        s.set(JAVA_INT, off(l, f), v);
    }

    public static void sl(MemorySegment s, GroupLayout l, String f, long v) {
        s.set(JAVA_LONG, off(l, f), v);
    }

    public static void sf(MemorySegment s, GroupLayout l, String f, float v) {
        s.set(JAVA_FLOAT, off(l, f), v);
    }

    public static void sa(MemorySegment s, GroupLayout l, String f, MemorySegment v) {
        s.set(ADDRESS, off(l, f), v);
    }

    public static int gi(MemorySegment s, GroupLayout l, String f) {
        return s.get(JAVA_INT, off(l, f));
    }

    public static long gl(MemorySegment s, GroupLayout l, String f) {
        return s.get(JAVA_LONG, off(l, f));
    }

    /** Invoke a command handle returning {@code VkResult} (or any int). Boxes args — fine off the hot path. */
    public static int invoke(MethodHandle h, Object... args) {
        try {
            return (int) h.invokeWithArguments(args);
        } catch (Throwable t) {
            throw NativeException.rethrow("vulkan call", t);
        }
    }

    /** Invoke a {@code void} command handle. */
    public static void invokeVoid(MethodHandle h, Object... args) {
        try {
            h.invokeWithArguments(args);
        } catch (Throwable t) {
            throw NativeException.rethrow("vulkan call", t);
        }
    }

    /**
     * Raise unless {@code result} is {@code VK_SUCCESS}.
     *
     * <p>Device loss gets its own type. Every other failure is one call reporting a fault with the device
     * still healthy; that one means the device and everything made from it is gone, and the difference decides
     * whether a caller retries or rebuilds. See {@link DeviceLostException}.
     */
    public static void check(int result, String call) {
        if (result == Vk.ERROR_DEVICE_LOST) {
            throw new DeviceLostException(call);
        }
        if (result != Vk.VK_SUCCESS) {
            throw new NativeException(call + " failed: VkResult " + result);
        }
    }
}
