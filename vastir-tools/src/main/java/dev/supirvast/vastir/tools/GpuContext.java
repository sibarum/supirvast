package dev.supirvast.vastir.tools;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkMappedMemoryRange;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import org.lwjgl.vulkan.EXTShaderAtomicFloat;
import org.lwjgl.vulkan.EXTShaderAtomicFloat2;
import org.lwjgl.vulkan.VkExtensionProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderAtomicFloatFeaturesEXT;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan11Properties;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan13Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan13Properties;
import org.lwjgl.vulkan.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo;
import dev.supirvast.vastir.lower.DeviceFeature;
import dev.supirvast.vastir.spirv.Capability;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.EnumSet;
import java.util.Set;

import static org.lwjgl.vulkan.VK13.*;

/**
 * A long-lived Vulkan compute context: the expensive-to-create objects (instance, device, queue, command
 * pool) are built once in {@link #open()} and held, and a {@link ResidentKernel} pipeline is built once per
 * kernel and dispatched against many times. This is the resident counterpart to a per-call setup — the closed
 * set of kernels is preloaded, and each {@link #dispatch} re-marshals only the data.
 *
 * <p>Headless compute only; requires Vulkan 1.3 (to consume SPIR-V 1.6). Per-dispatch storage buffers are
 * still allocated and freed each call (persistent/ring buffers are a later, streaming-oriented step); the win
 * here is eliminating instance/device/pipeline rebuild, which dominates the cost.
 *
 * <p><b>Concurrency.</b> {@link #dispatch} is synchronous (submit then wait). For overlapping work,
 * {@link #submitAsync} records + submits a dispatch against a fence <em>without</em> blocking and returns a
 * {@link Submission}; {@link #await} then blocks on that fence and reads the results back. Several
 * submissions can be in flight at once, distributed round-robin over the device's compute queues, so their
 * device execution (and host readback) overlaps. Recording is <em>not</em> internally synchronized: like
 * the rest of this context, {@code submitAsync}/{@code await} must be called from the owning thread (one
 * command pool, serialized recording); the overlap that buys you is on the <em>device</em>, across queues.
 * A given kernel has a single descriptor set, so it may have only <em>one</em> submission in flight at a
 * time — {@code submitAsync} throws if a second targets an already-pending kernel (distinct kernels run
 * concurrently freely). That is exactly enough for "launch N different kernels, then await all N".
 */
public final class GpuContext implements AutoCloseable {

    static {
        // LWJGL's VkInstance constructor eagerly builds its capability set by enumerating each physical
        // device's extensions onto the calling thread's off-heap MemoryStack. VkExtensionProperties is 260
        // bytes, so a device exposing ~250+ extensions (common on current GPU drivers) needs >64 KB and
        // overflows LWJGL's default 64 KB stack with "OutOfMemoryError: Out of stack space." — which surfaces
        // here as a build failure when the GPU tests run. Raise the per-thread stack well above that. This
        // runs at class initialization, before any static method below materializes a MemoryStack.
        Configuration.STACK_SIZE.set(512); // KiB per thread; default is 64
    }

    private static final int RESULT_BYTES = Integer.BYTES;

    /** Compute queues to request from the chosen family (capped by what it offers). More ⇒ more overlap. */
    private static final int MAX_QUEUES = 4;

    private final VkInstance instance;
    private final VkPhysicalDevice physical;
    private final VkDevice device;
    private final VkQueue[] queues;
    private int nextQueue;              // round-robin cursor; owning-thread only, no sync needed
    private final int queueFamily;
    private final long commandPool;
    private final Set<Capability> capabilities;
    private final long maxWorkgroupMemoryBytes;
    private final Set<DeviceFeature> features;
    private final String deviceName;
    private final String deviceType;
    private final int minSubgroupSize;
    private final int maxSubgroupSize;
    private final boolean subgroupSizeControl;

    private GpuContext(VkInstance instance, VkPhysicalDevice physical, VkDevice device, VkQueue[] queues,
            int queueFamily, long commandPool, Set<Capability> capabilities, long maxWorkgroupMemoryBytes,
            Set<DeviceFeature> features, String deviceName, String deviceType, Supported supported) {
        this.minSubgroupSize = supported.minSubgroupSize();
        this.maxSubgroupSize = supported.maxSubgroupSize();
        this.subgroupSizeControl = supported.subgroupSizeControl();
        this.deviceName = deviceName;
        this.deviceType = deviceType;
        this.instance = instance;
        this.physical = physical;
        this.device = device;
        this.queues = queues;
        this.queueFamily = queueFamily;
        this.commandPool = commandPool;
        this.capabilities = capabilities;
        this.maxWorkgroupMemoryBytes = maxWorkgroupMemoryBytes;
        this.features = features;
    }

    /** The SPIR-V capabilities this device supports (and that have been enabled on the logical device). */
    public Set<Capability> capabilities() {
        return capabilities;
    }

    /**
     * The most workgroup memory one compute pipeline may declare — the device's
     * {@code maxComputeSharedMemorySize}: 16 KB at least, 32–64 KB on current desktop hardware.
     */
    public long maxWorkgroupMemoryBytes() {
        return maxWorkgroupMemoryBytes;
    }

    /**
     * The device features this device supports (and that have been enabled) among those no capability
     * distinguishes — which kinds of memory its float atomics may point to.
     */
    public Set<DeviceFeature> features() {
        return features;
    }

    /**
     * Whether a Vulkan 1.3 device with a compute queue is usable on this machine.
     *
     * @throws IllegalStateException if {@code -Dsupirvast.gpu} names a device that is not there — a request
     *                               for one GPU is not answered by running on the CPU instead
     */
    public static boolean isAvailable() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            try {
                return pickComputeDevice(instance, stack) != null;
            } finally {
                vkDestroyInstance(instance, null);
            }
        } catch (NoSuchDevice unmatched) {
            throw unmatched;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** The name of the device this context runs on, as the driver reports it. */
    public String deviceName() {
        return deviceName;
    }

    /**
     * Whether a compute pipeline can require full subgroups of exactly {@code size} lanes here: size control
     * for compute, and a power of two in the device's {@code minSubgroupSize..maxSubgroupSize} (32 only, on
     * NVIDIA; 8 to 32 on current Intel).
     */
    public boolean supportsSubgroupSize(int size) {
        return subgroupSizeControl && Integer.bitCount(size) == 1 && size >= minSubgroupSize
                && size <= maxSubgroupSize;
    }

    /** The kind of device this context runs on: {@code discrete}, {@code integrated}, {@code virtual}, ... */
    public String deviceType() {
        return deviceType;
    }

    /** Creates the resident context (instance, device, queue, command pool). Caller must {@link #close()} it. */
    public static GpuContext open() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            VkPhysicalDevice physical = pickComputeDevice(instance, stack);
            if (physical == null) {
                vkDestroyInstance(instance, null);
                throw new IllegalStateException("no Vulkan compute device available");
            }
            int queueFamily = computeQueueFamily(physical, stack);
            int queueCount = Math.min(MAX_QUEUES, familyQueueCount(physical, queueFamily, stack));
            Supported supported = querySupported(physical, stack);
            VkDevice device = createDevice(physical, queueFamily, queueCount, supported, stack);
            VkQueue[] queues = deviceQueues(device, queueFamily, queueCount, stack);
            long commandPool = createCommandPool(device, queueFamily, stack);
            VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
            vkGetPhysicalDeviceProperties(physical, properties);
            long workgroupMemory = Integer.toUnsignedLong(properties.limits().maxComputeSharedMemorySize());
            return new GpuContext(instance, physical, device, queues, queueFamily, commandPool,
                    capabilitySet(supported), workgroupMemory, featureSet(supported),
                    properties.deviceNameString(), DeviceSelection.typeName(properties.deviceType()), supported);
        }
    }

    /**
     * Builds a resident pipeline for {@code spirv}'s {@code entryPoint} over {@code bindingCount} storage
     * buffers (descriptor set 0, bindings {@code 0..bindingCount-1}). Returned handle is reusable across many
     * dispatches and must be {@link ResidentKernel#close() closed} (before this context).
     */
    public ResidentKernel build(byte[] spirv, String entryPoint, int bindingCount) {
        return build(spirv, entryPoint, bindingCount, 1);
    }

    /**
     * As {@link #build(byte[], String, int)}, for a kernel whose workgroups are {@code workgroupSize}
     * invocations wide. Above one, the dispatch rounds up to whole workgroups, so the shader must deal with
     * the invocations past the requested count — stop them, or for a kernel with a barrier bound them itself.
     * Either way it reads that count ({@code Expr.InvocationCount}) from a 4-byte push constant at offset 0,
     * which every pipeline's layout declares and every dispatch sets; a shader that never reads it ignores it.
     */
    public ResidentKernel build(byte[] spirv, String entryPoint, int bindingCount, int workgroupSize) {
        return build(spirv, entryPoint, bindingCount, workgroupSize, 0);
    }

    /**
     * As {@link #build(byte[], String, int, int)}, for a kernel with subgroup operations that needs full
     * subgroups of exactly {@code subgroupSize} lanes — so what they compute is the kernel's to say, not the
     * device's. 0 asks for nothing, which is right for every kernel without them. The device must {@link
     * #supportsSubgroupSize support} the size and the workgroup must be a whole number of subgroups.
     */
    public ResidentKernel build(byte[] spirv, String entryPoint, int bindingCount, int workgroupSize,
            int subgroupSize) {
        if (workgroupSize < 1) {
            throw new IllegalArgumentException("workgroup size must be >= 1, got " + workgroupSize);
        }
        if (subgroupSize != 0 && (!supportsSubgroupSize(subgroupSize) || workgroupSize % subgroupSize != 0)) {
            throw new IllegalArgumentException("cannot require full subgroups of " + subgroupSize + " lanes in a "
                    + "workgroup of " + workgroupSize + " on " + deviceName + " (it offers " + minSubgroupSize
                    + ".." + maxSubgroupSize + (subgroupSizeControl ? "" : ", without size control") + ")");
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long shaderModule = createShaderModule(device, spirv, stack);
            long setLayout = createSetLayout(device, bindingCount, stack);
            long pipelineLayout = createPipelineLayout(device, setLayout, stack);
            long pipeline = createComputePipeline(device, pipelineLayout, shaderModule, entryPoint, subgroupSize,
                    stack);
            // The pipeline/layouts are immutable and safe to share across concurrent dispatches; the
            // mutable binding state (the descriptor set) is allocated PER submission instead, so one
            // pipeline can back several in-flight dispatches at once (see submitAsync).
            return new ResidentKernel(device, shaderModule, setLayout, pipelineLayout, pipeline, bindingCount,
                    workgroupSize);
        }
    }

    /**
     * Dispatches {@code kernel} over {@code invocations} invocations against the given buffers (one per
     * binding; inputs pre-filled, outputs sized) — as {@code ceil(invocations / workgroupSize)} workgroups.
     * Returns the buffers' contents read back after execution. Storage buffers are allocated and freed within
     * the call; the pipeline is reused.
     */
    public int[][] dispatch(ResidentKernel kernel, int[][] buffers, int invocations) {
        return await(submitAsync(kernel, buffers, invocations));
    }

    /**
     * Records and submits a dispatch against a fence <em>without</em> waiting, returning a
     * {@link Submission} for {@link #await}. Several submissions can be outstanding at once — distributed
     * round-robin over the compute queues — so their device execution and host readback overlap, including
     * several concurrent dispatches of the <em>same</em> pipeline (each gets its own descriptor set and
     * buffers). Must be called on the owning thread (recording is not internally synchronized).
     */
    public Submission submitAsync(ResidentKernel kernel, int[][] buffers, int invocations) {
        if (buffers.length != kernel.bindingCount) {
            throw new IllegalArgumentException("kernel expects " + kernel.bindingCount + " buffers, got "
                    + buffers.length);
        }
        int n = buffers.length;
        long[] bufferHandles = new long[n];
        long[] memoryHandles = new long[n];
        int[] lengths = new int[n];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            for (int i = 0; i < n; i++) {
                long size = Math.max(RESULT_BYTES, (long) buffers[i].length * Integer.BYTES);
                bufferHandles[i] = createBuffer(device, size, stack);
                memoryHandles[i] = allocateAndBind(physical, device, bufferHandles[i], stack);
                writeInts(device, memoryHandles[i], buffers[i]);
                lengths[i] = buffers[i].length;
            }
            // A fresh descriptor set PER submission (from a per-submission pool) — the mutable binding
            // state that must be independent for concurrent dispatches. Freed with its pool in await().
            long descriptorPool = createDescriptorPool(device, kernel.bindingCount, stack);
            long descriptorSet = allocateDescriptorSet(device, descriptorPool, kernel.setLayout, stack);
            bindBuffers(device, descriptorSet, bufferHandles, stack);

            VkCommandBuffer cmd = recordDispatch(device, commandPool, kernel.pipeline, kernel.pipelineLayout,
                    descriptorSet, kernel, invocations, stack);
            long fence = createFence(device, stack);
            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(cmd));
            check(vkQueueSubmit(nextQueue(), submit, fence), "vkQueueSubmit");
            return new Submission(cmd, fence, descriptorPool, bufferHandles, memoryHandles, lengths);
        }
    }

    /**
     * Blocks on {@code submission}'s fence, reads back its output buffers, and frees its per-dispatch
     * resources (fence, command buffer, descriptor pool, storage buffers and memory). Owning-thread only;
     * each submission is awaited exactly once.
     */
    public int[][] await(Submission submission) {
        if (submission.awaited) {
            throw new IllegalStateException("submission already awaited");
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            check(vkWaitForFences(device, stack.longs(submission.fence), true, Long.MAX_VALUE),
                    "vkWaitForFences");
        }
        int n = submission.bufferHandles.length;
        int[][] results = new int[n][];
        for (int i = 0; i < n; i++) {
            results[i] = readInts(device, submission.memoryHandles[i], submission.bufferLengths[i]);
        }
        vkDestroyFence(device, submission.fence, null);
        vkFreeCommandBuffers(device, commandPool, submission.cmd);
        vkDestroyDescriptorPool(device, submission.descriptorPool, null);
        for (int i = 0; i < n; i++) {
            vkFreeMemory(device, submission.memoryHandles[i], null);
            vkDestroyBuffer(device, submission.bufferHandles[i], null);
        }
        submission.awaited = true;
        return results;
    }

    /** Round-robins the compute queues so consecutive submissions can execute on different queues. */
    private VkQueue nextQueue() {
        VkQueue q = queues[nextQueue];
        nextQueue = (nextQueue + 1) % queues.length;
        return q;
    }

    // --- resident buffers ------------------------------------------------------------------------------
    //
    // Buffers that live in device-local memory between dispatches, so a stepped kernel pays for transfers
    // only when the host actually wants the data. Everything touching them goes through one queue, and every
    // command buffer on it opens with a memory barrier, so each dispatch or copy sees the writes of all the
    // work submitted before it -- a pipeline barrier's first scope is everything earlier in submission order
    // on the same queue, across command buffers. That is what lets a dispatch be submitted without waiting.

    /** Resident work submitted and not yet reclaimed; past this many, a dispatch waits for the oldest. */
    private static final int MAX_PENDING = 64;

    private final java.util.ArrayDeque<Pending> pending = new java.util.ArrayDeque<>();

    /** A submitted resident command buffer and what to free once its fence signals. */
    private record Pending(VkCommandBuffer cmd, long fence, long descriptorPool) {}

    /**
     * A storage buffer in device-local memory that outlives dispatches. Owning-thread only, like the rest of
     * this context; close it before the context.
     */
    public static final class DeviceBuffer implements AutoCloseable {
        private final GpuContext owner;
        private final long buffer;
        private final long memory;
        private final int words;
        private boolean closed;

        private DeviceBuffer(GpuContext owner, long buffer, long memory, int words) {
            this.owner = owner;
            this.buffer = buffer;
            this.memory = memory;
            this.words = words;
        }

        /** Capacity in 32-bit words. */
        public int words() {
            return words;
        }

        /** Waits for resident work that may still use it, then frees it. Idempotent. */
        @Override
        public void close() {
            if (!closed) {
                owner.finish();
                vkDestroyBuffer(owner.device, buffer, null);
                vkFreeMemory(owner.device, memory, null);
                closed = true;
            }
        }

        private long handle() {
            if (closed) {
                throw new IllegalStateException("device buffer is closed");
            }
            return buffer;
        }
    }

    /** A device-local buffer of {@code words} 32-bit words, contents undefined until written. */
    public DeviceBuffer allocateBuffer(int words) {
        if (words < 1) {
            throw new IllegalArgumentException("a device buffer needs at least one word, got " + words);
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long buffer = createBuffer(device, (long) words * Integer.BYTES,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT, stack);
            // Device-local where the implementation has it for this buffer; any allowed type otherwise, which
            // on an integrated GPU is the same memory anyway.
            Allocation allocation = allocate(physical, device, buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, stack);
            return new DeviceBuffer(this, buffer, allocation.memory(), words);
        }
    }

    /** Replaces the start of {@code target} with {@code data}, through a staging buffer. Waits for the copy. */
    public void write(DeviceBuffer target, int[] data) {
        if (data.length > target.words()) {
            throw new IllegalArgumentException(data.length + " words do not fit a " + target.words() + "-word buffer");
        }
        if (data.length == 0) {
            return;
        }
        long size = (long) data.length * Integer.BYTES;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long staging = createBuffer(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stack);
            Allocation memory = allocate(physical, device, staging, 0,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stack);
            try {
                writeInts(device, memory.memory(), data);
                VkCommandBuffer cmd = beginOneShot(stack);
                residentBarrier(cmd, stack);
                vkCmdCopyBuffer(cmd, staging, target.handle(), VkBufferCopy.calloc(1, stack).size(size));
                check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                submitResidentAndWait(cmd, stack);
            } finally {
                vkDestroyBuffer(device, staging, null);
                vkFreeMemory(device, memory.memory(), null);
            }
        }
    }

    /**
     * The first {@code words} words of {@code source}, once all resident work submitted before this call has
     * finished with it. Through a host-cached staging buffer where the device has one: an uncached read of
     * mapped memory is what made the per-dispatch path slow.
     */
    public int[] read(DeviceBuffer source, int words) {
        if (words > source.words()) {
            throw new IllegalArgumentException(words + " words exceed a " + source.words() + "-word buffer");
        }
        if (words == 0) {
            return new int[0];
        }
        long size = (long) words * Integer.BYTES;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long staging = createBuffer(device, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, stack);
            Allocation memory = allocate(physical, device, staging, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stack);
            try {
                VkCommandBuffer cmd = beginOneShot(stack);
                residentBarrier(cmd, stack);
                vkCmdCopyBuffer(cmd, source.handle(), staging, VkBufferCopy.calloc(1, stack).size(size));
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                        VkMemoryBarrier.calloc(1, stack).sType$Default()
                                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_HOST_READ_BIT),
                        null, null);
                check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                submitResidentAndWait(cmd, stack);
                if (!memory.coherent()) {
                    check(vkInvalidateMappedMemoryRanges(device, VkMappedMemoryRange.calloc(1, stack).sType$Default()
                            .memory(memory.memory()).offset(0).size(VK_WHOLE_SIZE)), "vkInvalidateMappedMemoryRanges");
                }
                return readInts(device, memory.memory(), words);
            } finally {
                vkDestroyBuffer(device, staging, null);
                vkFreeMemory(device, memory.memory(), null);
            }
        }
    }

    /**
     * Submits {@code kernel} over {@code invocations} invocations against resident {@code buffers} (one per
     * binding) and returns without waiting. The next dispatch, {@link #read} or {@link #write} on this context
     * is ordered after it. No data moves.
     */
    public void dispatchResident(ResidentKernel kernel, DeviceBuffer[] buffers, int invocations) {
        if (buffers.length != kernel.bindingCount) {
            throw new IllegalArgumentException("kernel expects " + kernel.bindingCount + " buffers, got "
                    + buffers.length);
        }
        if (pending.size() >= MAX_PENDING) {
            retire(pending.removeFirst(), true);
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long[] handles = new long[buffers.length];
            for (int i = 0; i < buffers.length; i++) {
                handles[i] = buffers[i].handle();
            }
            long descriptorPool = createDescriptorPool(device, kernel.bindingCount, stack);
            long descriptorSet = allocateDescriptorSet(device, descriptorPool, kernel.setLayout, stack);
            bindBuffers(device, descriptorSet, handles, stack);

            VkCommandBuffer cmd = beginOneShot(stack);
            residentBarrier(cmd, stack);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0,
                    stack.longs(descriptorSet), null);
            vkCmdPushConstants(cmd, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, stack.ints(invocations));
            vkCmdDispatch(cmd, kernel.groupsFor(invocations), 1, 1);
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

            long fence = createFence(device, stack);
            check(vkQueueSubmit(queues[0], VkSubmitInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(cmd)), fence), "vkQueueSubmit");
            pending.addLast(new Pending(cmd, fence, descriptorPool));
        }
        reclaim();
    }

    /** Blocks until every resident dispatch submitted so far has finished, and frees what they held. */
    public void finish() {
        while (!pending.isEmpty()) {
            retire(pending.removeFirst(), true);
        }
    }

    /** Frees the resident work that has already finished, oldest first, without waiting. */
    private void reclaim() {
        while (!pending.isEmpty() && vkGetFenceStatus(device, pending.peekFirst().fence()) == VK_SUCCESS) {
            retire(pending.removeFirst(), false);
        }
    }

    private void retire(Pending work, boolean wait) {
        if (wait) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                check(vkWaitForFences(device, stack.longs(work.fence()), true, Long.MAX_VALUE), "vkWaitForFences");
            }
        }
        vkDestroyFence(device, work.fence(), null);
        vkFreeCommandBuffers(device, commandPool, work.cmd());
        vkDestroyDescriptorPool(device, work.descriptorPool(), null);
    }

    private VkCommandBuffer beginOneShot(MemoryStack stack) {
        PointerBuffer pCmd = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO).commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1), pCmd), "vkAllocateCommandBuffers");
        VkCommandBuffer cmd = new VkCommandBuffer(pCmd.get(0), device);
        check(vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)), "vkBeginCommandBuffer");
        return cmd;
    }

    /** Submits a copy on the resident queue — ordered after every pending dispatch — and waits for it. */
    private void submitResidentAndWait(VkCommandBuffer cmd, MemoryStack stack) {
        long fence = createFence(device, stack);
        check(vkQueueSubmit(queues[0], VkSubmitInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                .pCommandBuffers(stack.pointers(cmd)), fence), "vkQueueSubmit");
        check(vkWaitForFences(device, stack.longs(fence), true, Long.MAX_VALUE), "vkWaitForFences");
        vkDestroyFence(device, fence, null);
        vkFreeCommandBuffers(device, commandPool, cmd);
        reclaim();   // the queue is in order, so everything submitted before this copy has finished too
    }

    /**
     * Every earlier shader or transfer write, made visible to every later shader or transfer access. Coarse on
     * purpose: one barrier per command buffer is negligible against a dispatch, and a finer one would have to
     * know which buffers each dispatch touches — which is the kind of bookkeeping that is wrong once.
     */
    private static void residentBarrier(VkCommandBuffer cmd, MemoryStack stack) {
        int stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(cmd, stages, stages, 0, VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                        | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT), null, null);
    }

    @Override
    public void close() {
        finish();
        vkDestroyCommandPool(device, commandPool, null);
        vkDestroyDevice(device, null);
        vkDestroyInstance(instance, null);
    }

    /**
     * A preloaded compute pipeline, reusable across dispatches — and across <em>concurrent</em> ones:
     * it holds only immutable objects (shader/pipeline/layouts), while each dispatch allocates its own
     * descriptor set and buffers in {@link #submitAsync}. Only {@link #setLayout} is read there, to
     * allocate those per-submission sets.
     */
    public static final class ResidentKernel implements AutoCloseable {
        private final VkDevice device;
        private final long shaderModule;
        private final long setLayout;
        private final long pipelineLayout;
        private final long pipeline;
        private final int bindingCount;
        private final int workgroupSize;

        ResidentKernel(VkDevice device, long shaderModule, long setLayout, long pipelineLayout, long pipeline,
                int bindingCount, int workgroupSize) {
            this.device = device;
            this.shaderModule = shaderModule;
            this.setLayout = setLayout;
            this.pipelineLayout = pipelineLayout;
            this.pipeline = pipeline;
            this.bindingCount = bindingCount;
            this.workgroupSize = workgroupSize;
        }

        /** Whole workgroups covering {@code invocations}; the shader deals with the ones past the end. */
        int groupsFor(int invocations) {
            return (int) (((long) invocations + workgroupSize - 1) / workgroupSize);
        }

        @Override
        public void close() {
            vkDestroyPipeline(device, pipeline, null);
            vkDestroyPipelineLayout(device, pipelineLayout, null);
            vkDestroyDescriptorSetLayout(device, setLayout, null);
            vkDestroyShaderModule(device, shaderModule, null);
        }
    }

    // --- Vulkan setup helpers (host-platform compute, no surface/swapchain) ----------------------------

    private static VkInstance createInstance(MemoryStack stack) {
        VkApplicationInfo app = VkApplicationInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                .pApplicationName(stack.UTF8("supir-vast"))
                .apiVersion(VK_API_VERSION_1_3);
        VkInstanceCreateInfo info = VkInstanceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                .pApplicationInfo(app);
        PointerBuffer pInstance = stack.mallocPointer(1);
        check(vkCreateInstance(info, null, pInstance), "vkCreateInstance");
        return new VkInstance(pInstance.get(0), info);
    }

    private static VkPhysicalDevice pickComputeDevice(VkInstance instance, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices");
        if (count.get(0) == 0) {
            return null;
        }
        PointerBuffer devices = stack.mallocPointer(count.get(0));
        check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices");
        java.util.List<DeviceSelection.Candidate> candidates = new java.util.ArrayList<>();
        VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.malloc(stack);
        for (int i = 0; i < devices.capacity(); i++) {
            VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
            vkGetPhysicalDeviceProperties(candidate, properties);
            candidates.add(new DeviceSelection.Candidate(i, properties.deviceNameString(), properties.deviceType(),
                    findComputeQueueFamily(candidate, stack) >= 0));
        }
        int chosen;
        try {
            chosen = DeviceSelection.choose(candidates, DeviceSelection.selector());
        } catch (IllegalStateException unmatched) {
            throw new NoSuchDevice(unmatched.getMessage());
        }
        return chosen < 0 ? null : new VkPhysicalDevice(devices.get(chosen), instance);
    }

    /** {@code -Dsupirvast.gpu} named a device that is not there; distinct so {@link #isAvailable} passes it on. */
    private static final class NoSuchDevice extends IllegalStateException {
        NoSuchDevice(String message) {
            super(message);
        }
    }

    private static int computeQueueFamily(VkPhysicalDevice device, MemoryStack stack) {
        int family = findComputeQueueFamily(device, stack);
        if (family < 0) {
            throw new IllegalStateException("device has no compute queue family");
        }
        return family;
    }

    /** How many queues the given family offers (≥ 1) — the ceiling on how much dispatch overlap we can get. */
    private static int familyQueueCount(VkPhysicalDevice device, int family, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, families);
        return families.get(family).queueCount();
    }

    private static int findComputeQueueFamily(VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, families);
        for (int i = 0; i < families.capacity(); i++) {
            if ((families.get(i).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) {
                return i;
            }
        }
        return -1;
    }

    private static VkDevice createDevice(VkPhysicalDevice physical, int queueFamily, int queueCount,
            Supported supported, MemoryStack stack) {
        float[] priorities = new float[queueCount];
        java.util.Arrays.fill(priorities, 1.0f);
        // pQueuePriorities(FloatBuffer) also sets queueCount to the buffer's remaining — so this requests
        // `queueCount` queues from the family (round-robined by submitAsync for overlap).
        VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                .queueFamilyIndex(queueFamily)
                .pQueuePriorities(stack.floats(priorities));
        // Enable exactly the supported features we may emit capabilities for (so emitting OpCapability Int64
        // etc. is actually licensed). Uses the Features2 pNext chain; pEnabledFeatures must then be null.
        VkPhysicalDeviceVulkan12Features features12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                .sType$Default().shaderInt8(supported.int8());
        // Size control and full subgroups are what let a kernel's subgroups be the ones it was written for.
        long chain = VkPhysicalDeviceVulkan13Features.calloc(stack).sType$Default().pNext(features12.address())
                .subgroupSizeControl(supported.subgroupSizeControl())
                .computeFullSubgroups(supported.subgroupSizeControl()).address();
        // The float-atomic extensions are enabled only when one of their features is, since enabling a feature
        // licenses what the lowering will then emit. float2 extends float, so min/max implies the first.
        java.util.List<String> extensions = new java.util.ArrayList<>();
        boolean float2 = supported.floatAtomicMinMax() || supported.sharedFloatAtomicMinMax();
        if (float2 || supported.floatAtomicAdd() || supported.sharedFloatAtomics() || supported.sharedFloatAtomicAdd()) {
            extensions.add(EXTShaderAtomicFloat.VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            chain = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack).sType$Default().pNext(chain)
                    .shaderBufferFloat32AtomicAdd(supported.floatAtomicAdd())
                    .shaderSharedFloat32Atomics(supported.sharedFloatAtomics())
                    .shaderSharedFloat32AtomicAdd(supported.sharedFloatAtomicAdd()).address();
        }
        if (float2) {
            extensions.add(EXTShaderAtomicFloat2.VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
            chain = VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT.calloc(stack).sType$Default().pNext(chain)
                    .shaderBufferFloat32AtomicMinMax(supported.floatAtomicMinMax())
                    .shaderSharedFloat32AtomicMinMax(supported.sharedFloatAtomicMinMax()).address();
        }
        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                .sType$Default().pNext(chain);
        features2.features()
                .shaderInt16(supported.int16())
                .shaderInt64(supported.int64())
                .shaderFloat64(supported.float64());
        PointerBuffer extensionNames = stack.mallocPointer(extensions.size());
        for (String extension : extensions) {
            extensionNames.put(stack.UTF8(extension));
        }
        extensionNames.flip();
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pNext(features2.address())
                .pQueueCreateInfos(queues)
                .ppEnabledExtensionNames(extensionNames);
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        return new VkDevice(pDevice.get(0), physical, info);
    }

    /**
     * What the physical device supports among the optional capabilities our lowering can emit.
     * {@code floatAtomicAdd}/{@code floatAtomicMinMax} are the storage-buffer features; the {@code shared}
     * ones license the same instructions on workgroup memory. {@code subgroupOperations} is the device's
     * {@code VkSubgroupFeatureFlags} for compute (0 if compute has none); {@code subgroupSizeControl} is size
     * control and full subgroups together, for the compute stage.
     */
    private record Supported(boolean int8, boolean int16, boolean int64, boolean float64,
            boolean floatAtomicAdd, boolean floatAtomicMinMax,
            boolean sharedFloatAtomics, boolean sharedFloatAtomicAdd, boolean sharedFloatAtomicMinMax,
            int subgroupOperations, int minSubgroupSize, int maxSubgroupSize, boolean subgroupSizeControl) {}

    private static Supported querySupported(VkPhysicalDevice physical, MemoryStack stack) {
        Set<String> extensions = deviceExtensions(physical, stack);
        boolean hasFloat = extensions.contains(EXTShaderAtomicFloat.VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        boolean hasFloat2 = hasFloat
                && extensions.contains(EXTShaderAtomicFloat2.VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);

        VkPhysicalDeviceVulkan12Features features12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default();
        VkPhysicalDeviceVulkan13Features features13 = VkPhysicalDeviceVulkan13Features.calloc(stack).sType$Default()
                .pNext(features12.address());
        long chain = features13.address();
        // A feature struct is chained only when its extension exists: querying one the driver does not know
        // is harmless in practice and undefined on paper.
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat = null;
        if (hasFloat) {
            atomicFloat = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack).sType$Default().pNext(chain);
            chain = atomicFloat.address();
        }
        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomicFloat2 = null;
        if (hasFloat2) {
            atomicFloat2 = VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT.calloc(stack).sType$Default().pNext(chain);
            chain = atomicFloat2.address();
        }
        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                .sType$Default().pNext(chain);
        vkGetPhysicalDeviceFeatures2(physical, features2);
        VkPhysicalDeviceFeatures core = features2.features();

        VkPhysicalDeviceVulkan11Properties properties11 = VkPhysicalDeviceVulkan11Properties.calloc(stack).sType$Default();
        VkPhysicalDeviceVulkan13Properties properties13 = VkPhysicalDeviceVulkan13Properties.calloc(stack).sType$Default()
                .pNext(properties11.address());
        vkGetPhysicalDeviceProperties2(physical,
                VkPhysicalDeviceProperties2.calloc(stack).sType$Default().pNext(properties13.address()));
        boolean compute = (properties11.subgroupSupportedStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        boolean sizeControl = features13.subgroupSizeControl() && features13.computeFullSubgroups()
                && (properties13.requiredSubgroupSizeStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0;

        return new Supported(features12.shaderInt8(), core.shaderInt16(), core.shaderInt64(), core.shaderFloat64(),
                atomicFloat != null && atomicFloat.shaderBufferFloat32AtomicAdd(),
                atomicFloat2 != null && atomicFloat2.shaderBufferFloat32AtomicMinMax(),
                atomicFloat != null && atomicFloat.shaderSharedFloat32Atomics(),
                atomicFloat != null && atomicFloat.shaderSharedFloat32AtomicAdd(),
                atomicFloat2 != null && atomicFloat2.shaderSharedFloat32AtomicMinMax(),
                compute ? properties11.subgroupSupportedOperations() : 0,
                properties13.minSubgroupSize(), properties13.maxSubgroupSize(), sizeControl);
    }

    /** The device features no capability distinguishes, as the lowering's target names them. */
    private static Set<DeviceFeature> featureSet(Supported s) {
        EnumSet<DeviceFeature> features = EnumSet.noneOf(DeviceFeature.class);
        if (s.floatAtomicAdd()) {
            features.add(DeviceFeature.BUFFER_FLOAT32_ATOMIC_ADD);
        }
        if (s.floatAtomicMinMax()) {
            features.add(DeviceFeature.BUFFER_FLOAT32_ATOMIC_MIN_MAX);
        }
        if (s.sharedFloatAtomics()) {
            features.add(DeviceFeature.SHARED_FLOAT32_ATOMICS);
        }
        if (s.sharedFloatAtomicAdd()) {
            features.add(DeviceFeature.SHARED_FLOAT32_ATOMIC_ADD);
        }
        if (s.sharedFloatAtomicMinMax()) {
            features.add(DeviceFeature.SHARED_FLOAT32_ATOMIC_MIN_MAX);
        }
        return Set.copyOf(features);
    }

    private static Set<String> deviceExtensions(VkPhysicalDevice physical, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumerateDeviceExtensionProperties(physical, (String) null, count, null),
                "vkEnumerateDeviceExtensionProperties");
        VkExtensionProperties.Buffer properties = VkExtensionProperties.malloc(count.get(0), stack);
        check(vkEnumerateDeviceExtensionProperties(physical, (String) null, count, properties),
                "vkEnumerateDeviceExtensionProperties");
        Set<String> names = new java.util.HashSet<>();
        for (int i = 0; i < properties.capacity(); i++) {
            names.add(properties.get(i).extensionNameString());
        }
        return names;
    }

    private static Set<Capability> capabilitySet(Supported s) {
        EnumSet<Capability> caps = EnumSet.of(Capability.Shader);
        if (s.int8()) {
            caps.add(Capability.Int8);
        }
        if (s.int16()) {
            caps.add(Capability.Int16);
        }
        if (s.int64()) {
            caps.add(Capability.Int64);
        }
        if (s.float64()) {
            caps.add(Capability.Float64);
        }
        // Subgroup operations in compute, by kind; each capability is core Vulkan 1.1 once its bit is set.
        int ops = s.subgroupOperations();
        if ((ops & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0) {
            caps.add(Capability.GroupNonUniform);
            if ((ops & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0) {
                caps.add(Capability.GroupNonUniformVote);
            }
            if ((ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0) {
                caps.add(Capability.GroupNonUniformArithmetic);
            }
            if ((ops & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0) {
                caps.add(Capability.GroupNonUniformShuffle);
            }
            if ((ops & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0) {
                caps.add(Capability.GroupNonUniformShuffleRelative);
            }
        }
        // One capability for both kinds of memory; which kinds the device licenses is its feature set's to say.
        if (s.floatAtomicAdd() || s.sharedFloatAtomicAdd()) {
            caps.add(Capability.AtomicFloat32AddEXT);
        }
        if (s.floatAtomicMinMax() || s.sharedFloatAtomicMinMax()) {
            caps.add(Capability.AtomicFloat32MinMaxEXT);
        }
        return Set.copyOf(caps);
    }

    private static VkQueue[] deviceQueues(VkDevice device, int queueFamily, int queueCount, MemoryStack stack) {
        VkQueue[] queues = new VkQueue[queueCount];
        PointerBuffer pQueue = stack.mallocPointer(1);
        for (int i = 0; i < queueCount; i++) {
            vkGetDeviceQueue(device, queueFamily, i, pQueue);
            queues[i] = new VkQueue(pQueue.get(0), device);
        }
        return queues;
    }

    private static long createBuffer(VkDevice device, long sizeBytes, MemoryStack stack) {
        return createBuffer(device, sizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, stack);
    }

    private static long createBuffer(VkDevice device, long sizeBytes, int usage, MemoryStack stack) {
        VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(sizeBytes)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        LongBuffer pBuffer = stack.mallocLong(1);
        check(vkCreateBuffer(device, info, null, pBuffer), "vkCreateBuffer");
        return pBuffer.get(0);
    }

    private static long allocateAndBind(VkPhysicalDevice physical, VkDevice device, long buffer, MemoryStack stack) {
        return allocate(physical, device, buffer, 0,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stack).memory();
    }

    /** Bound device memory, and whether host writes and reads of it need no flush or invalidate. */
    private record Allocation(long memory, boolean coherent) {}

    /**
     * Allocates and binds memory for {@code buffer} from a type that has every {@code required} property,
     * preferring one that also has every {@code preferred} property.
     */
    private static Allocation allocate(VkPhysicalDevice physical, VkDevice device, long buffer, int preferred,
            int required, MemoryStack stack) {
        VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, buffer, requirements);

        VkPhysicalDeviceMemoryProperties memProps = VkPhysicalDeviceMemoryProperties.malloc(stack);
        vkGetPhysicalDeviceMemoryProperties(physical, memProps);
        int typeIndex = memoryType(memProps, requirements.memoryTypeBits(), required | preferred);
        if (typeIndex < 0) {
            typeIndex = memoryType(memProps, requirements.memoryTypeBits(), required);
        }
        if (typeIndex < 0) {
            throw new IllegalStateException("no memory type with properties 0x" + Integer.toHexString(required)
                    + " for the buffer");
        }

        VkMemoryAllocateInfo info = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(requirements.size())
                .memoryTypeIndex(typeIndex);
        LongBuffer pMemory = stack.mallocLong(1);
        check(vkAllocateMemory(device, info, null, pMemory), "vkAllocateMemory");
        long memory = pMemory.get(0);
        check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        boolean coherent = (memProps.memoryTypes(typeIndex).propertyFlags() & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        return new Allocation(memory, coherent);
    }

    private static int memoryType(VkPhysicalDeviceMemoryProperties memProps, int allowedTypes, int properties) {
        for (int i = 0; i < memProps.memoryTypeCount(); i++) {
            boolean allowed = (allowedTypes & (1 << i)) != 0;
            if (allowed && (memProps.memoryTypes(i).propertyFlags() & properties) == properties) {
                return i;
            }
        }
        return -1;
    }

    private static long createShaderModule(VkDevice device, byte[] spirv, MemoryStack stack) {
        ByteBuffer code = MemoryUtil.memAlloc(spirv.length).put(spirv);
        code.flip();
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(vkCreateShaderModule(device, info, null, pModule), "vkCreateShaderModule");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }

    private static long createSetLayout(VkDevice device, int bindingCount, MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(bindingCount, stack);
        for (int i = 0; i < bindingCount; i++) {
            bindings.get(i)
                    .binding(i)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        }
        VkDescriptorSetLayoutCreateInfo info = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(bindings);
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreateDescriptorSetLayout(device, info, null, pLayout), "vkCreateDescriptorSetLayout");
        return pLayout.get(0);
    }

    /** Set 0, and the 4-byte invocation count at push-constant offset 0 that every dispatch sets. */
    private static long createPipelineLayout(VkDevice device, long setLayout, MemoryStack stack) {
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(setLayout))
                .pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(Integer.BYTES));
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreatePipelineLayout(device, info, null, pLayout), "vkCreatePipelineLayout");
        return pLayout.get(0);
    }

    /** @param subgroupSize full subgroups of exactly this many lanes, or 0 to leave it to the device */
    private static long createComputePipeline(VkDevice device, long layout, long shaderModule, String entryPoint,
            int subgroupSize, MemoryStack stack) {
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                .module(shaderModule)
                .pName(stack.UTF8(entryPoint));
        if (subgroupSize != 0) {
            // Both halves matter: the size fixes which invocations share a subgroup, and full subgroups mean
            // none of them is missing a lane — without which a reduction would quietly cover fewer values.
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required =
                    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.calloc(stack).sType$Default();
            // LWJGL 3.3.6 generates this field without a setter; it has the offset, so write it there.
            MemoryUtil.memPutInt(required.address()
                    + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.REQUIREDSUBGROUPSIZE, subgroupSize);
            stage.flags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT).pNext(required.address());
        }
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                .stage(stage)
                .layout(layout);
        LongBuffer pPipeline = stack.mallocLong(1);
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, info, null, pPipeline), "vkCreateComputePipelines");
        return pPipeline.get(0);
    }

    private static long createDescriptorPool(VkDevice device, int descriptorCount, MemoryStack stack) {
        VkDescriptorPoolSize.Buffer size = VkDescriptorPoolSize.calloc(1, stack)
                .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(descriptorCount);
        VkDescriptorPoolCreateInfo info = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                .maxSets(1)
                .pPoolSizes(size);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateDescriptorPool(device, info, null, pPool), "vkCreateDescriptorPool");
        return pPool.get(0);
    }

    private static long allocateDescriptorSet(VkDevice device, long pool, long setLayout, MemoryStack stack) {
        VkDescriptorSetAllocateInfo info = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(pool)
                .pSetLayouts(stack.longs(setLayout));
        LongBuffer pSet = stack.mallocLong(1);
        check(vkAllocateDescriptorSets(device, info, pSet), "vkAllocateDescriptorSets");
        return pSet.get(0);
    }

    private static void bindBuffers(VkDevice device, long descriptorSet, long[] buffers, MemoryStack stack) {
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(buffers.length, stack);
        for (int i = 0; i < buffers.length; i++) {
            VkDescriptorBufferInfo.Buffer info = VkDescriptorBufferInfo.calloc(1, stack)
                    .buffer(buffers[i]).offset(0).range(VK_WHOLE_SIZE);
            writes.get(i)
                    .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(descriptorSet)
                    .dstBinding(i)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(1)
                    .pBufferInfo(info);
        }
        vkUpdateDescriptorSets(device, writes, null);
    }

    private static void writeInts(VkDevice device, long memory, int[] data) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long size = Math.max(RESULT_BYTES, (long) data.length * Integer.BYTES);
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, size, 0, pData), "vkMapMemory");
            MemoryUtil.memByteBuffer(pData.get(0), (int) size).asIntBuffer().put(data);
            vkUnmapMemory(device, memory);
        }
    }

    private static int[] readInts(VkDevice device, long memory, int length) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long size = Math.max(RESULT_BYTES, (long) length * Integer.BYTES);
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, size, 0, pData), "vkMapMemory");
            int[] out = new int[length];
            MemoryUtil.memByteBuffer(pData.get(0), (int) size).asIntBuffer().get(out);
            vkUnmapMemory(device, memory);
            return out;
        }
    }

    private static long createCommandPool(VkDevice device, int queueFamily, MemoryStack stack) {
        VkCommandPoolCreateInfo info = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(queueFamily);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, info, null, pPool), "vkCreateCommandPool");
        return pPool.get(0);
    }

    private static VkCommandBuffer recordDispatch(
            VkDevice device, long commandPool, long pipeline, long pipelineLayout, long descriptorSet,
            ResidentKernel kernel, int invocations, MemoryStack stack) {
        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(1);
        PointerBuffer pCmd = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, allocInfo, pCmd), "vkAllocateCommandBuffers");
        VkCommandBuffer cmd = new VkCommandBuffer(pCmd.get(0), device);

        VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        check(vkBeginCommandBuffer(cmd, begin), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                stack.longs(descriptorSet), null);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, stack.ints(invocations));
        vkCmdDispatch(cmd, kernel.groupsFor(invocations), 1, 1);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        return cmd;
    }

    private static long createFence(VkDevice device, MemoryStack stack) {
        VkFenceCreateInfo info = VkFenceCreateInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        LongBuffer pFence = stack.mallocLong(1);
        check(vkCreateFence(device, info, null, pFence), "vkCreateFence");
        return pFence.get(0);
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
