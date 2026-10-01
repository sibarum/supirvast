package dev.supirvast.vulkan;

import dev.supirvast.ffi.NativeException;

/**
 * {@code VK_ERROR_DEVICE_LOST} — a GPU reset, a driver crash, a hot-unplugged eGPU.
 *
 * <p>Its own type because the recovery is different in kind from every other failed call. An ordinary
 * {@link NativeException} may be a bug in one call with the device still perfectly healthy; this one means
 * every {@code VkPipeline}, {@code VkBuffer}, {@code VkImage}, queue and command pool made from that device is
 * now invalid, no retry of anything helps, and the only way forward is to destroy the device and build a new
 * one. A caller that wants to tell those apart should not have to read an exception message to do it.
 *
 * <p>It is also what lets the engine publish {@code EngineEvents.DEVICE_LOST} truthfully: an event that could
 * only ever be raised by string-matching a message would be an event nobody could trust.
 */
public final class DeviceLostException extends NativeException {

    private final String call;

    public DeviceLostException(String call) {
        super(call + " failed: VK_ERROR_DEVICE_LOST — the device is gone; everything created from it is "
                + "invalid and must be rebuilt on a new device");
        this.call = call;
    }

    /** The native call that reported the loss — where, as well as what. */
    public String call() {
        return call;
    }
}
