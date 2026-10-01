package dev.supirvast.vulkan;

import dev.supirvast.ffi.Ffi;
import dev.supirvast.ffi.NativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * The Vulkan loader binding. Unlike a plain OS library (user32), Vulkan exposes almost nothing by symbol name —
 * you obtain {@code vkGetInstanceProcAddr} from the loader ({@code vulkan-1}) and then resolve every other
 * command through it: global commands with a {@code NULL} instance, instance/device commands with a live handle.
 * This is the sanctioned "loader-based library" variant of the binding convention (see
 * {@code docs/native-bindings.md} §4.4); like all bindings it routes through {@link Ffi}.
 *
 * <p>The loader library name is platform-specific ({@code vulkan-1} on Windows). Windows-first for now; a later
 * step sources the name from the active platform when Linux/macOS land.
 */
public final class VkLoader {

    private static final SymbolLookup LIB = Ffi.library("vulkan-1");

    /** {@code vkGetInstanceProcAddr} — the one command resolved by name; everything else goes through it. */
    private static final MemorySegment GET_INSTANCE_PROC_ADDR_PTR = LIB.find("vkGetInstanceProcAddr")
            .orElseThrow(() -> new NativeException("vkGetInstanceProcAddr not found in the Vulkan loader"));

    private static final MethodHandle GET_INSTANCE_PROC_ADDR = Ffi.downcall(LIB, "vkGetInstanceProcAddr",
            FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));

    private VkLoader() {
    }

    /**
     * The raw {@code vkGetInstanceProcAddr} pointer — handed to {@code dev.vexelray.os.NativeWindow#createVulkanSurface}
     * so the OS module can load its platform surface entry point without a Vulkan-binding dependency.
     */
    public static MemorySegment getInstanceProcAddrPointer() {
        return GET_INSTANCE_PROC_ADDR_PTR;
    }

    /** Resolve a command pointer for {@code name} scoped to {@code instance} ({@code NULL} for global commands). */
    public static MemorySegment procAddr(MemorySegment instance, String name) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment cName = temp.allocateFrom(name);
            MemorySegment fn = (MemorySegment) GET_INSTANCE_PROC_ADDR.invokeExact(instance, cName);
            if (fn.equals(MemorySegment.NULL)) {
                throw new NativeException("vkGetInstanceProcAddr returned NULL for " + name);
            }
            return fn;
        } catch (Throwable t) {
            throw NativeException.rethrow("vkGetInstanceProcAddr(" + name + ")", t);
        }
    }

    /** A downcall handle for a global command (resolved with a {@code NULL} instance). */
    public static MethodHandle globalCommand(String name, FunctionDescriptor descriptor) {
        return Ffi.downcall(procAddr(MemorySegment.NULL, name), descriptor);
    }

    /** A downcall handle for an instance-level command, resolved against {@code instance}. */
    public static MethodHandle instanceCommand(MemorySegment instance, String name, FunctionDescriptor descriptor) {
        return Ffi.downcall(procAddr(instance, name), descriptor);
    }
}
