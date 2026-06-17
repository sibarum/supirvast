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

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Executes a SPIR-V compute shader on a real Vulkan device and reads back a single {@code int} the shader
 * wrote to a storage buffer at descriptor set 0, binding 0 (the SSBO emitted by {@code CoreToSpirv} for a
 * {@code StoreResult}).
 *
 * <p>Headless — compute only, no surface/swapchain. Requires Vulkan 1.3 (to consume SPIR-V 1.6). Each call
 * builds and tears down its own instance/device; this is verification code, not a hot path.
 */
public final class VulkanCompute {

    private static final int RESULT_BYTES = Integer.BYTES;

    /** Whether a Vulkan 1.3 device with a compute queue is usable on this machine. */
    public boolean isAvailable() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            try {
                VkPhysicalDevice device = pickComputeDevice(instance, stack);
                return device != null;
            } finally {
                vkDestroyInstance(instance, null);
            }
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Runs the shader's {@code entryPoint} with a 1×1×1 dispatch and returns {@code buffer[0]}. */
    public int execute(byte[] spirv, String entryPoint) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            VkPhysicalDevice physical = pickComputeDevice(instance, stack);
            if (physical == null) {
                throw new IllegalStateException("no Vulkan compute device available");
            }
            int queueFamily = computeQueueFamily(physical, stack);

            VkDevice device = createDevice(physical, queueFamily, stack);
            VkQueue queue = deviceQueue(device, queueFamily, stack);

            long buffer = createBuffer(device, RESULT_BYTES, stack);
            long memory = allocateAndBind(physical, device, buffer, stack);
            long shaderModule = createShaderModule(device, spirv, stack);
            long setLayout = createDescriptorSetLayout(device, stack);
            long pipelineLayout = createPipelineLayout(device, setLayout, stack);
            long pipeline = createComputePipeline(device, pipelineLayout, shaderModule, entryPoint, stack);
            long descriptorPool = createDescriptorPool(device, stack);
            long descriptorSet = allocateDescriptorSet(device, descriptorPool, setLayout, stack);
            bindBufferToDescriptor(device, descriptorSet, buffer, stack);

            long commandPool = createCommandPool(device, queueFamily, stack);
            VkCommandBuffer cmd = recordDispatch(device, commandPool, pipeline, pipelineLayout, descriptorSet, 1, stack);
            submitAndWait(queue, cmd, stack);

            int result = readResult(device, memory);

            // Teardown (reverse order).
            vkDestroyCommandPool(device, commandPool, null);
            vkDestroyDescriptorPool(device, descriptorPool, null);
            vkDestroyPipeline(device, pipeline, null);
            vkDestroyPipelineLayout(device, pipelineLayout, null);
            vkDestroyDescriptorSetLayout(device, setLayout, null);
            vkDestroyShaderModule(device, shaderModule, null);
            vkFreeMemory(device, memory, null);
            vkDestroyBuffer(device, buffer, null);
            vkDestroyDevice(device, null);
            vkDestroyInstance(instance, null);
            return result;
        }
    }

    /**
     * Runs a data-parallel kernel. {@code buffers[i]} is the contents of the storage buffer at binding
     * {@code i} (inputs pre-filled, outputs sized and zeroed); the kernel is dispatched with
     * {@code groupCountX} workgroups. Returns the buffers' contents read back after execution.
     */
    public int[][] executeKernel(byte[] spirv, String entryPoint, int[][] buffers, int groupCountX) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkInstance instance = createInstance(stack);
            VkPhysicalDevice physical = pickComputeDevice(instance, stack);
            if (physical == null) {
                throw new IllegalStateException("no Vulkan compute device available");
            }
            int queueFamily = computeQueueFamily(physical, stack);
            VkDevice device = createDevice(physical, queueFamily, stack);
            VkQueue queue = deviceQueue(device, queueFamily, stack);

            int n = buffers.length;
            long[] bufferHandles = new long[n];
            long[] memoryHandles = new long[n];
            for (int i = 0; i < n; i++) {
                long size = Math.max(RESULT_BYTES, (long) buffers[i].length * Integer.BYTES);
                bufferHandles[i] = createBuffer(device, size, stack);
                memoryHandles[i] = allocateAndBind(physical, device, bufferHandles[i], stack);
                writeInts(device, memoryHandles[i], buffers[i]);
            }

            long shaderModule = createShaderModule(device, spirv, stack);
            long setLayout = createKernelSetLayout(device, n, stack);
            long pipelineLayout = createPipelineLayout(device, setLayout, stack);
            long pipeline = createComputePipeline(device, pipelineLayout, shaderModule, entryPoint, stack);
            long descriptorPool = createKernelDescriptorPool(device, n, stack);
            long descriptorSet = allocateDescriptorSet(device, descriptorPool, setLayout, stack);
            bindBuffersToDescriptor(device, descriptorSet, bufferHandles, stack);

            long commandPool = createCommandPool(device, queueFamily, stack);
            VkCommandBuffer cmd = recordDispatch(device, commandPool, pipeline, pipelineLayout, descriptorSet,
                    groupCountX, stack);
            submitAndWait(queue, cmd, stack);

            int[][] results = new int[n][];
            for (int i = 0; i < n; i++) {
                results[i] = readInts(device, memoryHandles[i], buffers[i].length);
            }

            vkDestroyCommandPool(device, commandPool, null);
            vkDestroyDescriptorPool(device, descriptorPool, null);
            vkDestroyPipeline(device, pipeline, null);
            vkDestroyPipelineLayout(device, pipelineLayout, null);
            vkDestroyDescriptorSetLayout(device, setLayout, null);
            vkDestroyShaderModule(device, shaderModule, null);
            for (int i = 0; i < n; i++) {
                vkFreeMemory(device, memoryHandles[i], null);
                vkDestroyBuffer(device, bufferHandles[i], null);
            }
            vkDestroyDevice(device, null);
            vkDestroyInstance(instance, null);
            return results;
        }
    }

    // --- setup steps -----------------------------------------------------------------------------------

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

    private static VkDevice createDevice(VkPhysicalDevice physical, int queueFamily, MemoryStack stack) {
        VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                .queueFamilyIndex(queueFamily)
                .pQueuePriorities(stack.floats(1.0f));
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pQueueCreateInfos(queues);
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        return new VkDevice(pDevice.get(0), physical, info);
    }

    private static VkQueue deviceQueue(VkDevice device, int queueFamily, MemoryStack stack) {
        PointerBuffer pQueue = stack.mallocPointer(1);
        vkGetDeviceQueue(device, queueFamily, 0, pQueue);
        return new VkQueue(pQueue.get(0), device);
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
            throw new IllegalStateException("no host-visible memory type for the output buffer");
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

    private static long createDescriptorSetLayout(VkDevice device, MemoryStack stack) {
        VkDescriptorSetLayoutBinding.Buffer binding = VkDescriptorSetLayoutBinding.calloc(1, stack)
                .binding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        VkDescriptorSetLayoutCreateInfo info = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                .pBindings(binding);
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

    private static long createDescriptorPool(VkDevice device, MemoryStack stack) {
        VkDescriptorPoolSize.Buffer size = VkDescriptorPoolSize.calloc(1, stack)
                .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1);
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

    private static void bindBufferToDescriptor(VkDevice device, long descriptorSet, long buffer, MemoryStack stack) {
        VkDescriptorBufferInfo.Buffer bufferInfo = VkDescriptorBufferInfo.calloc(1, stack)
                .buffer(buffer)
                .offset(0)
                .range(VK_WHOLE_SIZE);
        VkWriteDescriptorSet.Buffer write = VkWriteDescriptorSet.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                .dstSet(descriptorSet)
                .dstBinding(0)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .pBufferInfo(bufferInfo);
        vkUpdateDescriptorSets(device, write, null);
    }

    private static long createKernelSetLayout(VkDevice device, int bindingCount, MemoryStack stack) {
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

    private static long createKernelDescriptorPool(VkDevice device, int descriptorCount, MemoryStack stack) {
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

    private static void bindBuffersToDescriptor(VkDevice device, long descriptorSet, long[] buffers, MemoryStack stack) {
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

    private static void submitAndWait(VkQueue queue, VkCommandBuffer cmd, MemoryStack stack) {
        VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                .pCommandBuffers(stack.pointers(cmd));
        check(vkQueueSubmit(queue, submit, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    }

    private static int readResult(VkDevice device, long memory) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, RESULT_BYTES, 0, pData), "vkMapMemory");
            int value = MemoryUtil.memByteBuffer(pData.get(0), RESULT_BYTES).getInt(0);
            vkUnmapMemory(device, memory);
            return value;
        }
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
