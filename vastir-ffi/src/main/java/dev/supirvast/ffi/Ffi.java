package dev.supirvast.ffi;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * The one sanctioned entry to Panama. <strong>Every</strong> native binding routes through here — no binding
 * calls {@link Linker}, {@link SymbolLookup#libraryLookup}, or {@link Linker#upcallStub} directly. Concentrating
 * the FFM ceremony in a single audited class is what makes the binding convention foolproof and reviewable.
 *
 * <p>Native-image note: {@link #library} resolves libraries at <em>runtime</em> (never at image-build time), and
 * every downcall/upcall built here is a registrable foreign call. Binding classes must not invoke native code in
 * their static initializers. See {@code docs/native-bindings.md} §6.
 */
public final class Ffi {

    private static final Linker LINKER = Linker.nativeLinker();

    /** Process-lifetime arena for symbols and upcall stubs that live as long as the engine. */
    public static final Arena GLOBAL = Arena.ofShared();

    private Ffi() {
    }

    /**
     * Open a system library by bare name ({@code "user32"}, {@code "X11"}), mapping it to the platform's file
     * name ({@code user32.dll} / {@code libX11.so}). Resolved at runtime against the process's library path.
     */
    public static SymbolLookup library(String name) {
        return SymbolLookup.libraryLookup(System.mapLibraryName(name), GLOBAL);
    }

    /** A downcall handle for {@code symbol} in {@code lib}. Fails loudly if the symbol is absent. */
    public static MethodHandle downcall(SymbolLookup lib, String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = lib.find(symbol)
                .orElseThrow(() -> new NativeException("symbol not found: " + symbol));
        return LINKER.downcallHandle(address, descriptor);
    }

    /**
     * A downcall handle for a raw function-pointer address — for entry points resolved at runtime rather than by
     * symbol name, e.g. a Vulkan function returned by {@code vkGetInstanceProcAddr}. The address must be non-null.
     */
    public static MethodHandle downcall(MemorySegment functionAddress, FunctionDescriptor descriptor) {
        if (functionAddress == null || functionAddress.equals(MemorySegment.NULL)) {
            throw new NativeException("cannot bind a null function pointer");
        }
        return LINKER.downcallHandle(functionAddress, descriptor);
    }

    /**
     * A {@link VarHandle} for a named struct field, with the layout's base-offset coordinate bound to 0 so
     * callers use the simple {@code (MemorySegment, value)} form. Offsets come from the layout, never a literal —
     * the foolproof way to read/write native structs (see {@code docs/native-bindings.md} §4.3).
     */
    public static VarHandle field(MemoryLayout layout, String name) {
        VarHandle vh = layout.varHandle(PathElement.groupElement(name));
        return MethodHandles.insertCoordinates(vh, 1, 0L).withInvokeExactBehavior();
    }

    /**
     * An upcall stub — a Java callback the OS can invoke (e.g. a WndProc) — bound to {@code lifetime}. The
     * lifetime arena must outlive every native use of the stub; binding it to a confined arena closed early is a
     * use-after-free.
     */
    public static MemorySegment upcall(MethodHandles.Lookup lookup, Class<?> owner, String method,
                                       FunctionDescriptor descriptor, Arena lifetime) {
        try {
            MethodHandle target = lookup.findStatic(owner, method, descriptor.toMethodType());
            return LINKER.upcallStub(target, descriptor, lifetime);
        } catch (ReflectiveOperationException e) {
            throw new NativeException("no upcall target " + owner.getSimpleName() + "#" + method, e);
        }
    }
}
