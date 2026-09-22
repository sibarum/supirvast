package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.type.Type;

/**
 * A kernel column that stays where the kernel runs between dispatches — the shape of a stepped simulation,
 * which should pay for a transfer only when the host actually wants to look. Made by
 * {@link Accelerator#allocate} and run against with {@link KernelHandle#dispatch}.
 *
 * <p>On a machine with a GPU it is device-local memory, reached from the host only through {@link #write} and
 * {@link #read}, each of which is a staged copy ordered after every dispatch submitted before it. Without one
 * it is a plain array the CPU backend updates in place. The two behave the same to a caller: that is the
 * point, since code written against resident buffers must still run where there is no device.
 *
 * <p>Data rides the same {@code int[]} wire as {@link KernelHandle#run}: one word per 32-bit element, two per
 * 64-bit element, low word first. Owning-thread only; close before the {@link Accelerator}, or let it close
 * whatever is left.
 */
public final class ResidentBuffer implements AutoCloseable {

    private final Type element;
    private final int elements;
    private final GpuContext context;          // null when host-resident
    private final GpuContext.DeviceBuffer device;
    private int[] host;
    private boolean closed;

    private ResidentBuffer(Type element, int elements, GpuContext context, GpuContext.DeviceBuffer device,
            int[] host) {
        this.element = element;
        this.elements = elements;
        this.context = context;
        this.device = device;
        this.host = host;
    }

    static ResidentBuffer onDevice(GpuContext context, Type element, int elements) {
        return new ResidentBuffer(element, elements, context,
                context.allocateBuffer(elements * wordsPerElement(element)), null);
    }

    static ResidentBuffer onHost(Type element, int elements) {
        return new ResidentBuffer(element, elements, null, null, new int[elements * wordsPerElement(element)]);
    }

    public Type element() {
        return element;
    }

    public int elements() {
        return elements;
    }

    /** Whether it lives in device memory, as opposed to a host array the CPU backend runs over. */
    public boolean onDevice() {
        return device != null;
    }

    /** Replaces the start of the buffer with {@code words}. */
    public void write(int[] words) {
        requireOpen();
        if (words.length > words()) {
            throw new IllegalArgumentException(words.length + " words do not fit a buffer of " + elements + " "
                    + element + " (" + words() + " words)");
        }
        if (device != null) {
            context.write(device, words);
        } else {
            System.arraycopy(words, 0, host, 0, words.length);
        }
    }

    /** The whole buffer, once every dispatch submitted before this call has finished with it. */
    public int[] read() {
        requireOpen();
        return device != null ? context.read(device, words()) : host.clone();
    }

    @Override
    public void close() {
        if (!closed) {
            if (device != null) {
                device.close();
            }
            host = null;
            closed = true;
        }
    }

    int words() {
        return elements * wordsPerElement(element);
    }

    GpuContext.DeviceBuffer device() {
        requireOpen();
        return device;
    }

    /** The live array, for the CPU backend to run over in place; host-resident buffers only. */
    int[] hostArray() {
        requireOpen();
        return host;
    }

    boolean isClosed() {
        return closed;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("resident buffer is closed");
        }
    }

    static int wordsPerElement(Type element) {
        int width = element instanceof Type.Int i ? i.width() : element instanceof Type.Float f ? f.width() : 32;
        return width == 64 ? 2 : 1;
    }
}
