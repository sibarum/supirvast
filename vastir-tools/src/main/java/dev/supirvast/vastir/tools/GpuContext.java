package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.lower.DeviceFeature;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vulkan.ComputeSupport;
import dev.supirvast.vulkan.VulkanDevice;
import dev.supirvast.vulkan.VulkanInstance;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

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
 * <h2>Whose device</h2>
 *
 * <p>{@link #open()} makes a device of its own and closes it. {@link #on} runs on one somebody else made — a
 * window's, so that what a simulation computes and what a renderer draws can be the same buffer, which two
 * devices cannot share. A borrowed device must have been made with compute support
 * ({@link VulkanDevice.Request#compute}), because that is what licenses the instructions a kernel emits, and it
 * is left open when this closes. Either way a {@code VkQueue} must be externally synchronised: this context
 * submits from its owning thread, so whoever shares the queue does so from that thread too, or under a lock the
 * two agree on.
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
 *
 * <h2>Threads</h2>
 *
 * <p>Two calls are not the owning thread's: {@link #build} and {@link #allocateBuffer} may be made from any
 * thread, at the same time as each other and as the owning thread's work. They only make objects, and Vulkan
 * does not ask for that to be synchronised; what does need it — the command pool, the queue and the record of
 * work in flight — they never touch. They are the slow part of setting up: a pipeline is the driver compiling a
 * shader, and an application that shares its device with a window cannot afford that on the thread that draws.
 * A kernel or buffer made elsewhere is handed to the owning thread before it is dispatched, written, read or
 * closed, and from then on is the owning thread's like any other.
 *
 * <p>Everything else — every dispatch, recording, submission, copy and wait — is the owning thread's.
 */
public final class GpuContext implements AutoCloseable {

    /** Compute queues to request from the chosen family (capped by what it offers). More ⇒ more overlap. */
    private static final int MAX_QUEUES = 4;

    private static final String APPLICATION_NAME = "supir-vast";

    /** Null when the device is borrowed: whoever made the instance closes it. */
    private final VulkanInstance instance;
    private final VulkanDevice device;
    private final boolean ownsDevice;
    private final VkCompute vk;
    private final List<MemorySegment> queues;
    private int nextQueue;              // round-robin cursor; owning-thread only, no sync needed
    private final long commandPool;
    private final Set<Capability> capabilities;
    private final long maxWorkgroupMemoryBytes;
    private final Set<DeviceFeature> features;
    private final String deviceName;
    private final String deviceType;
    private final int minSubgroupSize;
    private final int maxSubgroupSize;
    private final boolean subgroupSizeControl;

    private GpuContext(VulkanInstance instance, VulkanDevice device, boolean ownsDevice, ComputeSupport support,
            String deviceName, String deviceType) {
        this.instance = instance;
        this.device = device;
        this.ownsDevice = ownsDevice;
        this.vk = new VkCompute(device);
        this.queues = device.queues();
        this.commandPool = vk.createCommandPool(device.queueFamilyIndex());
        this.capabilities = capabilitySet(support);
        this.maxWorkgroupMemoryBytes = support.maxWorkgroupMemoryBytes();
        this.features = featureSet(support);
        this.deviceName = deviceName;
        this.deviceType = deviceType;
        this.minSubgroupSize = support.minSubgroupSize();
        this.maxSubgroupSize = support.maxSubgroupSize();
        this.subgroupSizeControl = support.subgroupSizeControl();
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
        try (VulkanInstance probe = new VulkanInstance(APPLICATION_NAME, List.of())) {
            return pickComputeDevice(probe) != null;
        } catch (NoSuchDevice unmatched) {
            throw unmatched;
        } catch (RuntimeException | LinkageError e) {
            // No loader, no driver, or an instance that will not make: this machine cannot, which is the answer.
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
        VulkanInstance instance = new VulkanInstance(APPLICATION_NAME, List.of());
        try {
            VulkanInstance.DeviceInfo info = pickComputeDevice(instance);
            if (info == null) {
                throw new IllegalStateException("no Vulkan compute device available");
            }
            ComputeSupport support = ComputeSupport.query(instance, info.physicalDevice());
            int queueCount = Math.min(MAX_QUEUES, info.computeQueueCount());
            VulkanDevice device = new VulkanDevice(instance.handle(), instance.selectionFor(info),
                    VulkanDevice.Request.headlessCompute(support, queueCount));
            try {
                return new GpuContext(instance, device, true, support, info.name(),
                        DeviceSelection.typeName(info.type()));
            } catch (RuntimeException | Error e) {
                device.close();
                throw e;
            }
        } catch (RuntimeException | Error e) {
            instance.close();
            throw e;
        }
    }

    /**
     * A context on a device somebody else made, so that what it computes lives where they draw.
     *
     * <p>The device must have been made with compute support, and exactly that support is what this context
     * reports and may use. It is left open by {@link #close()}, and so is {@code instance}: close the context
     * first, then the device and the instance, as they were made.
     *
     * @param instance the instance {@code device} was made from; asked what the device is called and what kind
     * @throws IllegalArgumentException if the device was made for drawing alone
     */
    public static GpuContext on(VulkanInstance instance, VulkanDevice device) {
        ComputeSupport support = device.computeSupport().orElseThrow(() -> new IllegalArgumentException(
                "the device was made without compute support, so it is not licensed for the instructions a "
                        + "kernel emits; make it with VulkanDevice.Request.presentAndCompute"));
        VulkanInstance.DeviceInfo info = instance.deviceInfos().stream()
                .filter(d -> d.physicalDevice().equals(device.physicalDevice())).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("the device is not on that instance"));
        // The queue the device was made with is the one every dispatch goes to, so it is that family that has to
        // run compute — not merely some family of the device.
        if (!instance.queueFamilySupports(device.physicalDevice(), device.queueFamilyIndex(),
                VulkanInstance.QUEUE_COMPUTE)) {
            throw new IllegalArgumentException("the device's queue family " + device.queueFamilyIndex()
                    + " on " + info.name() + " cannot run compute, so kernels cannot be dispatched to its queue");
        }
        return new GpuContext(null, device, false, support, info.name(), DeviceSelection.typeName(info.type()));
    }

    /**
     * Builds a resident pipeline for {@code spirv}'s {@code entryPoint} over {@code bindingCount} storage
     * buffers (descriptor set 0, bindings {@code 0..bindingCount-1}). Returned handle is reusable across many
     * dispatches and must be {@link ResidentKernel#close() closed} (before this context).
     *
     * <p>Any thread may build, at once with any other; see the class.
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
        long shaderModule = vk.createShaderModule(spirv);
        long setLayout = vk.createSetLayout(bindingCount);
        long pipelineLayout = vk.createPipelineLayout(setLayout);
        long pipeline = vk.createComputePipeline(pipelineLayout, shaderModule, entryPoint, subgroupSize);
        // The pipeline/layouts are immutable and safe to share across concurrent dispatches; the
        // mutable binding state (the descriptor set) is allocated PER submission instead, so one
        // pipeline can back several in-flight dispatches at once (see submitAsync).
        return new ResidentKernel(vk, shaderModule, setLayout, pipelineLayout, pipeline, bindingCount,
                workgroupSize);
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
        for (int i = 0; i < n; i++) {
            long size = Math.max(Integer.BYTES, (long) buffers[i].length * Integer.BYTES);
            bufferHandles[i] = vk.createBuffer(size, VkCompute.BUFFER_USAGE_STORAGE);
            memoryHandles[i] = vk.allocateHostVisible(bufferHandles[i]);
            vk.writeInts(memoryHandles[i], buffers[i]);
            lengths[i] = buffers[i].length;
        }
        // A fresh descriptor set PER submission (from a per-submission pool) — the mutable binding
        // state that must be independent for concurrent dispatches. Freed with its pool in await().
        long descriptorPool = vk.createDescriptorPool(kernel.bindingCount, 1);
        long descriptorSet = vk.allocateDescriptorSet(descriptorPool, kernel.setLayout);
        vk.bindBuffers(descriptorSet, bufferHandles);

        MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_ONE_TIME_SUBMIT);
        vk.recordDispatch(cmd, kernel.pipeline, kernel.pipelineLayout, descriptorSet, invocations,
                kernel.groupsFor(invocations));
        vk.endCommandBuffer(cmd);
        long fence = vk.createFence();
        vk.submit(nextQueue(), cmd, fence);
        return new Submission(cmd, fence, descriptorPool, bufferHandles, memoryHandles, lengths);
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
        vk.waitFence(submission.fence);
        int n = submission.bufferHandles.length;
        int[][] results = new int[n][];
        for (int i = 0; i < n; i++) {
            results[i] = vk.readInts(submission.memoryHandles[i], submission.bufferLengths[i]);
        }
        vk.destroyFence(submission.fence);
        vk.freeCommandBuffer(commandPool, submission.cmd);
        vk.destroyDescriptorPool(submission.descriptorPool);
        for (int i = 0; i < n; i++) {
            vk.freeMemory(submission.memoryHandles[i]);
            vk.destroyBuffer(submission.bufferHandles[i]);
        }
        submission.awaited = true;
        return results;
    }

    /** Round-robins the compute queues so consecutive submissions can execute on different queues. */
    private MemorySegment nextQueue() {
        MemorySegment q = queues.get(nextQueue);
        nextQueue = (nextQueue + 1) % queues.size();
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

    /**
     * A submitted resident command buffer and what to free once its fence signals — both null/0 for a run of a
     * {@link RecordedSequence}, which owns them itself.
     */
    private record Pending(MemorySegment cmd, long fence, long descriptorPool) {}

    /**
     * A storage buffer in device-local memory that outlives dispatches. {@linkplain #allocateBuffer Made} on any
     * thread, and from then on the owning thread's, like the rest of this context; close it before the context.
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

        /**
         * The {@code VkBuffer}, so that something drawing on the same device can bind it and read what the
         * kernels wrote where it already is. Made with storage-buffer usage.
         *
         * <p>Still this buffer's: it is freed by {@link #close()}, and whatever binds it must be done first.
         * Ordering the kernels that write it against a draw that reads it is the caller's: wait for the context's
         * work ({@link GpuContext#finish}) before submitting the draw.
         */
        public long vkBuffer() {
            return handle();
        }

        /** Waits for resident work that may still use it, then frees it. Idempotent. */
        @Override
        public void close() {
            if (!closed) {
                owner.finish();
                owner.vk.destroyBuffer(buffer);
                owner.vk.freeMemory(memory);
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

    /**
     * A device-local buffer of {@code words} 32-bit words, contents undefined until written or {@linkplain #clear
     * cleared}. Any thread may allocate; see the class.
     */
    public DeviceBuffer allocateBuffer(int words) {
        if (words < 1) {
            throw new IllegalArgumentException("a device buffer needs at least one word, got " + words);
        }
        long buffer = vk.createBuffer((long) words * Integer.BYTES, VkCompute.BUFFER_USAGE_STORAGE
                | VkCompute.BUFFER_USAGE_TRANSFER_SRC | VkCompute.BUFFER_USAGE_TRANSFER_DST);
        // Device-local where the implementation has it for this buffer; any allowed type otherwise, which
        // on an integrated GPU is the same memory anyway.
        VkCompute.Allocation allocation = vk.allocate(buffer, VkCompute.MEMORY_DEVICE_LOCAL, 0);
        return new DeviceBuffer(this, buffer, allocation.memory(), words);
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
        long staging = vk.createBuffer(size, VkCompute.BUFFER_USAGE_TRANSFER_SRC);
        VkCompute.Allocation memory = vk.allocate(staging, 0,
                VkCompute.MEMORY_HOST_VISIBLE | VkCompute.MEMORY_HOST_COHERENT);
        try {
            vk.writeInts(memory.memory(), data);
            MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_ONE_TIME_SUBMIT);
            vk.residentBarrier(cmd);
            vk.recordCopy(cmd, staging, target.handle(), size);
            vk.endCommandBuffer(cmd);
            submitResidentAndWait(cmd);
        } finally {
            vk.destroyBuffer(staging);
            vk.freeMemory(memory.memory());
        }
    }

    /**
     * Sets every word of every one of {@code targets} to zero, in one submission, and waits for it: what a fresh
     * buffer is when its kernels count on starting from nothing. Nothing comes from the host, so it costs what the
     * device takes to write the memory, where {@link #write} of zeros would stage a buffer as large and copy it.
     * Ordered after every dispatch submitted before it, like a write.
     */
    public void clear(List<DeviceBuffer> targets) {
        if (targets.isEmpty()) {
            return;
        }
        MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_ONE_TIME_SUBMIT);
        vk.residentBarrier(cmd);
        for (DeviceBuffer target : targets) {
            vk.recordFill(cmd, target.handle(), 0);
        }
        vk.endCommandBuffer(cmd);
        submitResidentAndWait(cmd);
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
        long staging = vk.createBuffer(size, VkCompute.BUFFER_USAGE_TRANSFER_DST);
        VkCompute.Allocation memory = vk.allocate(staging, VkCompute.MEMORY_HOST_CACHED,
                VkCompute.MEMORY_HOST_VISIBLE);
        try {
            MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_ONE_TIME_SUBMIT);
            vk.residentBarrier(cmd);
            vk.recordCopy(cmd, source.handle(), staging, size);
            vk.transferToHostBarrier(cmd);
            vk.endCommandBuffer(cmd);
            submitResidentAndWait(cmd);
            if (!memory.coherent()) {
                vk.invalidate(memory.memory());
            }
            return vk.readInts(memory.memory(), words);
        } finally {
            vk.destroyBuffer(staging);
            vk.freeMemory(memory.memory());
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
        long[] handles = new long[buffers.length];
        for (int i = 0; i < buffers.length; i++) {
            handles[i] = buffers[i].handle();
        }
        long descriptorPool = vk.createDescriptorPool(kernel.bindingCount, 1);
        long descriptorSet = vk.allocateDescriptorSet(descriptorPool, kernel.setLayout);
        vk.bindBuffers(descriptorSet, handles);

        MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_ONE_TIME_SUBMIT);
        vk.residentBarrier(cmd);
        vk.recordDispatch(cmd, kernel.pipeline, kernel.pipelineLayout, descriptorSet, invocations,
                kernel.groupsFor(invocations));
        vk.endCommandBuffer(cmd);

        long fence = vk.createFence();
        vk.submit(queues.get(0), cmd, fence);
        pending.addLast(new Pending(cmd, fence, descriptorPool));
        reclaim();
    }

    /** One dispatch of a {@link RecordedSequence}: a pipeline, its buffers in binding order, and a count. */
    public record Step(ResidentKernel kernel, DeviceBuffer[] buffers, int invocations) {
        public Step {
            if (buffers.length != kernel.bindingCount) {
                throw new IllegalArgumentException("kernel expects " + kernel.bindingCount + " buffers, got "
                        + buffers.length);
            }
            buffers = buffers.clone();
        }
    }

    /**
     * Records {@code steps} into one command buffer, to be {@link #submit submitted} as often as wanted. Each
     * step's descriptor set is allocated and written here, once, and its invocation count recorded as its push
     * constant, so a run costs one queue submission however many dispatches it holds. The command buffer opens
     * with the resident barrier, as every resident one does, and has the same barrier between consecutive
     * steps: each dispatch sees everything the previous one wrote, and cannot overwrite what it still reads.
     *
     * <p>The buffers and pipelines are referenced, not owned: they must outlive the sequence, and a sequence
     * must not be submitted after any of them is closed.
     */
    public RecordedSequence record(List<Step> steps) {
        if (steps.isEmpty()) {
            throw new IllegalArgumentException("a sequence needs at least one dispatch");
        }
        int descriptors = steps.stream().mapToInt(s -> s.kernel().bindingCount).sum();
        long descriptorPool = vk.createDescriptorPool(descriptors, steps.size());
        long[] sets = new long[steps.size()];
        for (int i = 0; i < sets.length; i++) {
            Step step = steps.get(i);
            sets[i] = vk.allocateDescriptorSet(descriptorPool, step.kernel().setLayout);
            long[] handles = new long[step.buffers().length];
            for (int b = 0; b < handles.length; b++) {
                handles[b] = step.buffers()[b].handle();
            }
            vk.bindBuffers(sets[i], handles);
        }

        // Simultaneous use: a run may be submitted again while the last one is still executing. The
        // descriptor sets are only read, so the runs can share them; the opening barrier orders them.
        MemorySegment cmd = vk.beginCommandBuffer(commandPool, VkCompute.COMMAND_BUFFER_SIMULTANEOUS_USE);
        for (int i = 0; i < sets.length; i++) {
            Step step = steps.get(i);
            vk.residentBarrier(cmd);
            vk.recordDispatch(cmd, step.kernel().pipeline, step.kernel().pipelineLayout, sets[i],
                    step.invocations(), step.kernel().groupsFor(step.invocations()));
        }
        vk.endCommandBuffer(cmd);
        return new RecordedSequence(this, cmd, descriptorPool, steps.size());
    }

    /**
     * Submits a recorded sequence's dispatches in one submission, without waiting. Like {@link
     * #dispatchResident}, it is ordered after all resident work submitted before it and before all after.
     */
    public void submit(RecordedSequence sequence) {
        if (sequence.closed) {
            throw new IllegalStateException("the sequence is closed");
        }
        if (pending.size() >= MAX_PENDING) {
            retire(pending.removeFirst(), true);
        }
        long fence = vk.createFence();
        vk.submit(queues.get(0), sequence.cmd, fence);
        pending.addLast(new Pending(null, fence, 0L));   // the sequence keeps its own
        reclaim();
    }

    /**
     * Dispatches recorded once into one command buffer, reused for every run: the per-dispatch cost left is
     * a share of one submission. Must be {@link #close closed} before the context, and before any pipeline or
     * buffer it records is.
     */
    public static final class RecordedSequence implements AutoCloseable {
        private final GpuContext context;
        private final MemorySegment cmd;
        private final long descriptorPool;
        private final int dispatches;
        private boolean closed;

        private RecordedSequence(GpuContext context, MemorySegment cmd, long descriptorPool, int dispatches) {
            this.context = context;
            this.cmd = cmd;
            this.descriptorPool = descriptorPool;
            this.dispatches = dispatches;
        }

        /** How many dispatches one run submits. */
        public int dispatches() {
            return dispatches;
        }

        /** Waits for every run in flight — the command buffer and sets are in use until then — and frees them. */
        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            context.finish();
            context.vk.freeCommandBuffer(context.commandPool, cmd);
            context.vk.destroyDescriptorPool(descriptorPool);
        }
    }

    /** Blocks until every resident dispatch submitted so far has finished, and frees what they held. */
    public void finish() {
        while (!pending.isEmpty()) {
            retire(pending.removeFirst(), true);
        }
    }

    /** Frees the resident work that has already finished, oldest first, without waiting. */
    private void reclaim() {
        while (!pending.isEmpty() && vk.fenceSignaled(pending.peekFirst().fence())) {
            retire(pending.removeFirst(), false);
        }
    }

    private void retire(Pending work, boolean wait) {
        if (wait) {
            vk.waitFence(work.fence());
        }
        vk.destroyFence(work.fence());
        if (work.cmd() != null) {   // a recorded sequence's run leaves its command buffer and sets to it
            vk.freeCommandBuffer(commandPool, work.cmd());
            vk.destroyDescriptorPool(work.descriptorPool());
        }
    }

    /** Submits a copy on the resident queue — ordered after every pending dispatch — and waits for it. */
    private void submitResidentAndWait(MemorySegment cmd) {
        long fence = vk.createFence();
        vk.submit(queues.get(0), cmd, fence);
        vk.waitFence(fence);
        vk.destroyFence(fence);
        vk.freeCommandBuffer(commandPool, cmd);
        reclaim();   // the queue is in order, so everything submitted before this copy has finished too
    }

    /**
     * Waits for this context's work and releases what it made. A borrowed device, and the instance it came
     * from, are left as they were.
     */
    @Override
    public void close() {
        finish();
        vk.destroyCommandPool(commandPool);
        if (ownsDevice) {
            device.close();
            instance.close();
        }
    }

    /**
     * A preloaded compute pipeline, reusable across dispatches — and across <em>concurrent</em> ones:
     * it holds only immutable objects (shader/pipeline/layouts), while each dispatch allocates its own
     * descriptor set and buffers in {@link #submitAsync}. Only {@link #setLayout} is read there, to
     * allocate those per-submission sets.
     */
    public static final class ResidentKernel implements AutoCloseable {
        private final VkCompute vk;
        private final long shaderModule;
        private final long setLayout;
        private final long pipelineLayout;
        private final long pipeline;
        private final int bindingCount;
        private final int workgroupSize;

        ResidentKernel(VkCompute vk, long shaderModule, long setLayout, long pipelineLayout, long pipeline,
                int bindingCount, int workgroupSize) {
            this.vk = vk;
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
            vk.destroyPipeline(pipeline);
            vk.destroyPipelineLayout(pipelineLayout);
            vk.destroySetLayout(setLayout);
            vk.destroyShaderModule(shaderModule);
        }
    }

    // --- device choice and what it supports ------------------------------------------------------------

    /** The device to run on, by the one policy ({@link DeviceSelection}); null when there is none to run on. */
    private static VulkanInstance.DeviceInfo pickComputeDevice(VulkanInstance instance) {
        List<VulkanInstance.DeviceInfo> infos = instance.deviceInfos();
        if (infos.isEmpty()) {
            return null;
        }
        List<DeviceSelection.Candidate> candidates = new ArrayList<>();
        for (VulkanInstance.DeviceInfo info : infos) {
            candidates.add(new DeviceSelection.Candidate(info.index(), info.name(), info.type(), info.compute()));
        }
        int chosen;
        try {
            chosen = DeviceSelection.choose(candidates, DeviceSelection.selector());
        } catch (IllegalStateException unmatched) {
            throw new NoSuchDevice(unmatched.getMessage());
        }
        return chosen < 0 ? null : infos.get(chosen);
    }

    /** {@code -Dsupirvast.gpu} named a device that is not there; distinct so {@link #isAvailable} passes it on. */
    private static final class NoSuchDevice extends IllegalStateException {
        NoSuchDevice(String message) {
            super(message);
        }
    }

    /** The device features no capability distinguishes, as the lowering's target names them. */
    private static Set<DeviceFeature> featureSet(ComputeSupport s) {
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

    private static Set<Capability> capabilitySet(ComputeSupport s) {
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
        if ((ops & ComputeSupport.SUBGROUP_BASIC) != 0) {
            caps.add(Capability.GroupNonUniform);
            if ((ops & ComputeSupport.SUBGROUP_VOTE) != 0) {
                caps.add(Capability.GroupNonUniformVote);
            }
            if ((ops & ComputeSupport.SUBGROUP_ARITHMETIC) != 0) {
                caps.add(Capability.GroupNonUniformArithmetic);
            }
            if ((ops & ComputeSupport.SUBGROUP_SHUFFLE) != 0) {
                caps.add(Capability.GroupNonUniformShuffle);
            }
            if ((ops & ComputeSupport.SUBGROUP_SHUFFLE_RELATIVE) != 0) {
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
}
