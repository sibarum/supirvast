package dev.supirvast.vulkan;

import dev.supirvast.ffi.Ffi;
import dev.supirvast.ffi.NativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * A logical {@code VkDevice} over a chosen {@link VulkanInstance.DeviceSelection}, with one graphics+present
 * queue and the {@code VK_KHR_swapchain} device extension enabled. Device-level commands are resolved through
 * {@code vkGetDeviceProcAddr} — {@link #command} exposes that resolver so higher-level Vulkan objects (images,
 * command buffers, the swapchain) bind exactly the commands they need. Destroys the device on {@link #close()}.
 */
public final class VulkanDevice implements AutoCloseable {

    private static final int VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
    private static final int VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
    private static final int VK_SUCCESS = 0;

    private static final GroupLayout DEVICE_QUEUE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"),
            JAVA_INT.withName("queueFamilyIndex"),
            JAVA_INT.withName("queueCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pQueuePriorities")
    ).withName("VkDeviceQueueCreateInfo");

    private static final GroupLayout DEVICE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"),
            JAVA_INT.withName("queueCreateInfoCount"),
            ADDRESS.withName("pQueueCreateInfos"),
            JAVA_INT.withName("enabledLayerCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("ppEnabledLayerNames"),
            JAVA_INT.withName("enabledExtensionCount"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("ppEnabledExtensionNames"),
            ADDRESS.withName("pEnabledFeatures")
    ).withName("VkDeviceCreateInfo");

    private static final GroupLayout MEMORY_TYPE = MemoryLayout.structLayout(
            JAVA_INT.withName("propertyFlags"), JAVA_INT.withName("heapIndex"));
    private static final GroupLayout MEMORY_HEAP = MemoryLayout.structLayout(
            JAVA_LONG.withName("size"), JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4));
    private static final GroupLayout MEMORY_PROPERTIES = MemoryLayout.structLayout(
            JAVA_INT.withName("memoryTypeCount"),
            MemoryLayout.sequenceLayout(32, MEMORY_TYPE).withName("memoryTypes"),
            JAVA_INT.withName("memoryHeapCount"),
            MemoryLayout.sequenceLayout(16, MEMORY_HEAP).withName("memoryHeaps")
    ).withName("VkPhysicalDeviceMemoryProperties");
    private static final long MEMORY_TYPES_OFFSET =
            MEMORY_PROPERTIES.byteOffset(PathElement.groupElement("memoryTypes"));
    private static final long MEMORY_TYPE_STRIDE = MEMORY_TYPE.byteSize();

    private static final VarHandle QCI_sType = Ffi.field(DEVICE_QUEUE_CREATE_INFO, "sType");
    private static final VarHandle QCI_queueFamilyIndex = Ffi.field(DEVICE_QUEUE_CREATE_INFO, "queueFamilyIndex");
    private static final VarHandle QCI_queueCount = Ffi.field(DEVICE_QUEUE_CREATE_INFO, "queueCount");
    private static final VarHandle QCI_pQueuePriorities = Ffi.field(DEVICE_QUEUE_CREATE_INFO, "pQueuePriorities");

    private static final VarHandle DCI_sType = Ffi.field(DEVICE_CREATE_INFO, "sType");
    private static final VarHandle DCI_queueCreateInfoCount = Ffi.field(DEVICE_CREATE_INFO, "queueCreateInfoCount");
    private static final VarHandle DCI_pQueueCreateInfos = Ffi.field(DEVICE_CREATE_INFO, "pQueueCreateInfos");
    private static final VarHandle DCI_enabledExtensionCount = Ffi.field(DEVICE_CREATE_INFO, "enabledExtensionCount");
    private static final VarHandle DCI_ppEnabledExtensionNames = Ffi.field(DEVICE_CREATE_INFO, "ppEnabledExtensionNames");

    private static final VarHandle MP_memoryTypeCount = Ffi.field(MEMORY_PROPERTIES, "memoryTypeCount");

    private static final VarHandle DCI_pNext = Ffi.field(DEVICE_CREATE_INFO, "pNext");

    private static final int VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x0004;

    /**
     * What to make a device with.
     *
     * @param swapchain  enable the {@code VK_KHR_swapchain} device extension. <b>False for a headless
     *                   device</b>, and not merely as a saving: that extension is only permitted on an
     *                   instance that enabled {@code VK_KHR_surface}, and a headless instance enables no
     *                   extensions at all. Asking anyway is
     *                   {@code VUID-vkCreateDevice-ppEnabledExtensionNames-01387} — which every driver here
     *                   honours by creating the device regardless, so the fault is invisible until something
     *                   turns the validation layer on
     * @param compute    the compute features to enable, as {@link ComputeSupport#query} found them, or null for
     *                   none. A device that will also run kernels has to be made with them: enabling a feature
     *                   is what licenses the instructions that use it, and it cannot be done afterwards
     * @param queueCount how many queues to take from the family, at least one. The caller caps it at what the
     *                   family offers. More queues let submissions overlap on the device; they do not make a
     *                   queue safe to use from two threads
     */
    public record Request(boolean swapchain, ComputeSupport compute, int queueCount) {

        public Request {
            if (queueCount < 1) {
                throw new IllegalArgumentException("a device needs at least one queue, got " + queueCount);
            }
        }

        /** A device that can present, and draws: what a windowed run needs. */
        public static Request present() {
            return new Request(true, null, 1);
        }

        /** A device that can present and also run compute kernels, which is what lets the two share it. */
        public static Request presentAndCompute(ComputeSupport support) {
            return new Request(true, support, 1);
        }

        /** A headless device for compute alone. */
        public static Request headlessCompute(ComputeSupport support, int queueCount) {
            return new Request(false, support, queueCount);
        }
    }

    private final MemorySegment handle;
    private final MemorySegment physicalDevice;
    private final java.util.List<MemorySegment> queues;
    private final ComputeSupport compute;
    private final int queueFamilyIndex;
    private final int maxPushConstantBytes;
    private final MethodHandle vkGetDeviceProcAddr;
    private final MethodHandle vkGetPhysicalDeviceMemoryProperties;
    private final MethodHandle vkDeviceWaitIdle;
    private final MethodHandle vkDestroyDevice;

    /** A device that can present: {@code VK_KHR_swapchain} enabled, which is what a windowed run needs. */
    public VulkanDevice(MemorySegment instance, VulkanInstance.DeviceSelection selection) {
        this(instance, selection, Request.present());
    }

    /** @param swapchain see {@link Request#swapchain} */
    public VulkanDevice(MemorySegment instance, VulkanInstance.DeviceSelection selection, boolean swapchain) {
        this(instance, selection, new Request(swapchain, null, 1));
    }

    public VulkanDevice(MemorySegment instance, VulkanInstance.DeviceSelection selection, Request request) {
        this.physicalDevice = selection.physicalDevice();
        this.queueFamilyIndex = selection.queueFamilyIndex();
        this.maxPushConstantBytes = selection.maxPushConstantBytes();
        this.compute = request.compute();

        MethodHandle vkCreateDevice = VkLoader.instanceCommand(instance, "vkCreateDevice",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        this.vkGetDeviceProcAddr = VkLoader.instanceCommand(instance, "vkGetDeviceProcAddr",
                FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
        this.vkGetPhysicalDeviceMemoryProperties = VkLoader.instanceCommand(instance,
                "vkGetPhysicalDeviceMemoryProperties", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

        try (Arena temp = Arena.ofConfined()) {
            int queueCount = request.queueCount();
            MemorySegment queueInfo = temp.allocate(DEVICE_QUEUE_CREATE_INFO);
            QCI_sType.set(queueInfo, VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
            QCI_queueFamilyIndex.set(queueInfo, queueFamilyIndex);
            QCI_queueCount.set(queueInfo, queueCount);
            MemorySegment priorities = temp.allocate(JAVA_FLOAT, queueCount);
            for (int q = 0; q < queueCount; q++) {
                priorities.setAtIndex(JAVA_FLOAT, q, 1.0f);
            }
            QCI_pQueuePriorities.set(queueInfo, priorities);

            java.util.List<String> extensions = new java.util.ArrayList<>();
            if (request.swapchain()) {
                extensions.add("VK_KHR_swapchain");
            }
            // The features chain is how they are enabled, and pEnabledFeatures must then stay null: the two are
            // alternatives, and naming both is a validation error.
            MemorySegment features = MemorySegment.NULL;
            if (compute != null) {
                ComputeSupport.Enabled enabled = compute.enable(temp);
                features = enabled.features2();
                extensions.addAll(enabled.extensions());
            }

            // Not a ternary at the set() below: VarHandle.set is signature-polymorphic and reads the static
            // type of its argument, so a conditional expression arrives as Object and fails at runtime.
            MemorySegment extArray = MemorySegment.NULL;
            if (!extensions.isEmpty()) {
                extArray = temp.allocate(ADDRESS, extensions.size());
                for (int e = 0; e < extensions.size(); e++) {
                    extArray.setAtIndex(ADDRESS, e, temp.allocateFrom(extensions.get(e)));
                }
            }

            MemorySegment createInfo = temp.allocate(DEVICE_CREATE_INFO);
            DCI_sType.set(createInfo, VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
            DCI_pNext.set(createInfo, features);
            DCI_queueCreateInfoCount.set(createInfo, 1);
            DCI_pQueueCreateInfos.set(createInfo, queueInfo);
            DCI_enabledExtensionCount.set(createInfo, extensions.size());
            DCI_ppEnabledExtensionNames.set(createInfo, extArray);

            MemorySegment pDevice = temp.allocate(ADDRESS);
            int result;
            try {
                result = (int) vkCreateDevice.invokeExact(selection.physicalDevice(), createInfo,
                        MemorySegment.NULL, pDevice);
            } catch (Throwable t) {
                throw NativeException.rethrow("vkCreateDevice", t);
            }
            if (result != VK_SUCCESS) {
                throw new NativeException("vkCreateDevice failed: VkResult " + result);
            }
            this.handle = pDevice.get(ADDRESS, 0);
        }

        MethodHandle vkGetDeviceQueue = command("vkGetDeviceQueue",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
        this.vkDeviceWaitIdle = command("vkDeviceWaitIdle", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        this.vkDestroyDevice = command("vkDestroyDevice", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

        java.util.List<MemorySegment> taken = new java.util.ArrayList<>();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pQueue = temp.allocate(ADDRESS);
            for (int q = 0; q < request.queueCount(); q++) {
                try {
                    vkGetDeviceQueue.invokeExact(handle, queueFamilyIndex, q, pQueue);
                } catch (Throwable t) {
                    throw NativeException.rethrow("vkGetDeviceQueue", t);
                }
                taken.add(pQueue.get(ADDRESS, 0));
            }
        }
        this.queues = java.util.List.copyOf(taken);
    }

    /** Resolve and bind a device-level command through {@code vkGetDeviceProcAddr}. Callers cache the result. */
    public MethodHandle command(String name, FunctionDescriptor descriptor) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment cName = temp.allocateFrom(name);
            MemorySegment fn = (MemorySegment) vkGetDeviceProcAddr.invokeExact(handle, cName);
            if (fn.equals(MemorySegment.NULL)) {
                throw new NativeException("vkGetDeviceProcAddr returned NULL for " + name);
            }
            return Ffi.downcall(fn, descriptor);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkGetDeviceProcAddr(" + name + ")", t);
        }
    }

    /**
     * The index of a memory type present in {@code typeBits} (from a {@code VkMemoryRequirements}) that has all of
     * {@code requiredProperties} (VK_MEMORY_PROPERTY_* flags). Throws if none qualifies.
     */
    public int findMemoryType(int typeBits, int requiredProperties) {
        int found = tryFindMemoryType(typeBits, requiredProperties);
        if (found < 0) {
            throw new NativeException("no memory type for typeBits=" + typeBits + " properties="
                    + requiredProperties);
        }
        return found;
    }

    /** As {@link #findMemoryType}, but -1 when there is none — for a caller with a second choice to try. */
    public int tryFindMemoryType(int typeBits, int requiredProperties) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = memoryProperties(temp);
            int count = (int) MP_memoryTypeCount.get(props);
            for (int i = 0; i < count; i++) {
                int flags = props.get(JAVA_INT, MEMORY_TYPES_OFFSET + (long) i * MEMORY_TYPE_STRIDE);
                if ((typeBits & (1 << i)) != 0 && (flags & requiredProperties) == requiredProperties) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Whether host writes and reads of memory of this type need no flush or invalidate — which decides whether
     * a mapped range has to be invalidated before the host reads what the device wrote.
     */
    public boolean memoryTypeIsHostCoherent(int typeIndex) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment props = memoryProperties(temp);
            int flags = props.get(JAVA_INT, MEMORY_TYPES_OFFSET + (long) typeIndex * MEMORY_TYPE_STRIDE);
            return (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        }
    }

    private MemorySegment memoryProperties(Arena arena) {
        MemorySegment props = arena.allocate(MEMORY_PROPERTIES);
        try {
            vkGetPhysicalDeviceMemoryProperties.invokeExact(physicalDevice, props);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkGetPhysicalDeviceMemoryProperties", t);
        }
        return props;
    }

    public MemorySegment handle() {
        return handle;
    }

    public MemorySegment physicalDevice() {
        return physicalDevice;
    }

    /** The first queue: the only one a device made for drawing has. */
    public MemorySegment queue() {
        return queues.get(0);
    }

    /**
     * Every queue taken from the family, in the order they were asked for.
     *
     * <p>A {@code VkQueue} must be externally synchronised, so sharing one device between drawing and
     * computing means sharing its queues: whatever submits to one does so from a single thread, or under a
     * lock the two agree on. Nothing here enforces that.
     */
    public java.util.List<MemorySegment> queues() {
        return queues;
    }

    /**
     * The compute features this device was made with — what {@link Request#compute} was given — or empty for a
     * device made for drawing alone. A kernel runner sharing a device reads this rather than querying afresh:
     * it must not assume more than the device was licensed for.
     */
    public java.util.Optional<ComputeSupport> computeSupport() {
        return java.util.Optional.ofNullable(compute);
    }

    public int queueFamilyIndex() {
        return queueFamilyIndex;
    }

    /**
     * What this device reports for {@code maxPushConstantsSize}, in bytes — at least 128 on anything Vulkan.
     *
     * <p>Here so that whoever composes a shader can take the cheaper road while it fits, and report headroom
     * rather than discovering the ceiling by hitting it. <b>Nothing that decides what a design may contain
     * may read it</b>: a design must open on a machine smaller than the one it was drawn on, and it does —
     * the values simply travel by the other road there.
     */
    public int maxPushConstantBytes() {
        return maxPushConstantBytes;
    }

    /** Block until the device has finished all submitted work. */
    public void waitIdle() {
        try {
            int r = (int) vkDeviceWaitIdle.invokeExact(handle);
            if (r != VK_SUCCESS) {
                throw new NativeException("vkDeviceWaitIdle failed: VkResult " + r);
            }
        } catch (Throwable t) {
            throw NativeException.rethrow("vkDeviceWaitIdle", t);
        }
    }

    @Override
    public void close() {
        try {
            vkDestroyDevice.invokeExact(handle, MemorySegment.NULL);
        } catch (Throwable t) {
            throw NativeException.rethrow("vkDestroyDevice", t);
        }
    }
}
