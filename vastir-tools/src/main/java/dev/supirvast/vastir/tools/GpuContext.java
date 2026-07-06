package dev.supirvast.vastir.tools;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCreateInfo;
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
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
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

    private GpuContext(VkInstance instance, VkPhysicalDevice physical, VkDevice device, VkQueue[] queues,
            int queueFamily, long commandPool, Set<Capability> capabilities) {
        this.instance = instance;
        this.physical = physical;
        this.device = device;
        this.queues = queues;
        this.queueFamily = queueFamily;
        this.commandPool = commandPool;
        this.capabilities = capabilities;
    }

    /** The SPIR-V capabilities this device supports (and that have been enabled on the logical device). */
    public Set<Capability> capabilities() {
        return capabilities;
    }

    /** Whether a Vulkan 1.3 device with a compute queue is usable on this machine. */
    public static boolean isAvailable() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            try {
                return pickComputeDevice(instance, stack) != null;
            } finally {
                vkDestroyInstance(instance, null);
            }
        } catch (RuntimeException e) {
            return false;
        }
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
            return new GpuContext(instance, physical, device, queues, queueFamily, commandPool,
                    capabilitySet(supported));
        }
    }

    /**
     * Builds a resident pipeline for {@code spirv}'s {@code entryPoint} over {@code bindingCount} storage
     * buffers (descriptor set 0, bindings {@code 0..bindingCount-1}). Returned handle is reusable across many
     * dispatches and must be {@link ResidentKernel#close() closed} (before this context).
     */
    public ResidentKernel build(byte[] spirv, String entryPoint, int bindingCount) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long shaderModule = createShaderModule(device, spirv, stack);
            long setLayout = createSetLayout(device, bindingCount, stack);
            long pipelineLayout = createPipelineLayout(device, setLayout, stack);
            long pipeline = createComputePipeline(device, pipelineLayout, shaderModule, entryPoint, stack);
            // The pipeline/layouts are immutable and safe to share across concurrent dispatches; the
            // mutable binding state (the descriptor set) is allocated PER submission instead, so one
            // pipeline can back several in-flight dispatches at once (see submitAsync).
            return new ResidentKernel(device, shaderModule, setLayout, pipelineLayout, pipeline, bindingCount);
        }
    }

    /**
     * Dispatches {@code kernel} with {@code groupCountX} workgroups against the given buffers (one per
     * binding; inputs pre-filled, outputs sized). Returns the buffers' contents read back after execution.
     * Storage buffers are allocated and freed within the call; the pipeline is reused.
     */
    public int[][] dispatch(ResidentKernel kernel, int[][] buffers, int groupCountX) {
        return await(submitAsync(kernel, buffers, groupCountX));
    }

    /**
     * Records and submits a dispatch against a fence <em>without</em> waiting, returning a
     * {@link Submission} for {@link #await}. Several submissions can be outstanding at once — distributed
     * round-robin over the compute queues — so their device execution and host readback overlap, including
     * several concurrent dispatches of the <em>same</em> pipeline (each gets its own descriptor set and
     * buffers). Must be called on the owning thread (recording is not internally synchronized).
     */
    public Submission submitAsync(ResidentKernel kernel, int[][] buffers, int groupCountX) {
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
                    descriptorSet, groupCountX, stack);
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

    @Override
    public void close() {
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

        ResidentKernel(VkDevice device, long shaderModule, long setLayout, long pipelineLayout, long pipeline,
                int bindingCount) {
            this.device = device;
            this.shaderModule = shaderModule;
            this.setLayout = setLayout;
            this.pipelineLayout = pipelineLayout;
            this.pipeline = pipeline;
            this.bindingCount = bindingCount;
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
        for (int i = 0; i < devices.capacity(); i++) {
            VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
            if (findComputeQueueFamily(candidate, stack) >= 0) {
                return candidate;
            }
        }
        return null;
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
        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                .sType$Default().pNext(features12.address());
        features2.features()
                .shaderInt16(supported.int16())
                .shaderInt64(supported.int64())
                .shaderFloat64(supported.float64());
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pNext(features2.address())
                .pQueueCreateInfos(queues);
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        return new VkDevice(pDevice.get(0), physical, info);
    }

    /** What the physical device supports among the optional capabilities our lowering can emit. */
    private record Supported(boolean int8, boolean int16, boolean int64, boolean float64) {}

    private static Supported querySupported(VkPhysicalDevice physical, MemoryStack stack) {
        VkPhysicalDeviceVulkan12Features features12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default();
        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack)
                .sType$Default().pNext(features12.address());
        vkGetPhysicalDeviceFeatures2(physical, features2);
        VkPhysicalDeviceFeatures core = features2.features();
        return new Supported(features12.shaderInt8(), core.shaderInt16(), core.shaderInt64(), core.shaderFloat64());
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
        VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(sizeBytes)
                .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        LongBuffer pBuffer = stack.mallocLong(1);
        check(vkCreateBuffer(device, info, null, pBuffer), "vkCreateBuffer");
        return pBuffer.get(0);
    }

    private static long allocateAndBind(VkPhysicalDevice physical, VkDevice device, long buffer, MemoryStack stack) {
        VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, buffer, requirements);

        VkPhysicalDeviceMemoryProperties memProps = VkPhysicalDeviceMemoryProperties.malloc(stack);
        vkGetPhysicalDeviceMemoryProperties(physical, memProps);
        int wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        int typeIndex = -1;
        for (int i = 0; i < memProps.memoryTypeCount(); i++) {
            boolean allowed = (requirements.memoryTypeBits() & (1 << i)) != 0;
            boolean visible = (memProps.memoryTypes(i).propertyFlags() & wanted) == wanted;
            if (allowed && visible) {
                typeIndex = i;
                break;
            }
        }
        if (typeIndex < 0) {
            throw new IllegalStateException("no host-visible memory type for the storage buffer");
        }

        VkMemoryAllocateInfo info = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(requirements.size())
                .memoryTypeIndex(typeIndex);
        LongBuffer pMemory = stack.mallocLong(1);
        check(vkAllocateMemory(device, info, null, pMemory), "vkAllocateMemory");
        long memory = pMemory.get(0);
        check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        return memory;
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

    private static long createPipelineLayout(VkDevice device, long setLayout, MemoryStack stack) {
        VkPipelineLayoutCreateInfo info = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pSetLayouts(stack.longs(setLayout));
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreatePipelineLayout(device, info, null, pLayout), "vkCreatePipelineLayout");
        return pLayout.get(0);
    }

    private static long createComputePipeline(
            VkDevice device, long layout, long shaderModule, String entryPoint, MemoryStack stack) {
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                .module(shaderModule)
                .pName(stack.UTF8(entryPoint));
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
            int groupCountX, MemoryStack stack) {
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
        vkCmdDispatch(cmd, groupCountX, 1, 1);
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
