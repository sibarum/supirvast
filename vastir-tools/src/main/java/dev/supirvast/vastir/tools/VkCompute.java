package dev.supirvast.vastir.tools;

import dev.supirvast.vulkan.Ffm;
import dev.supirvast.vulkan.VulkanDevice;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static dev.supirvast.vulkan.Ffm.gi;
import static dev.supirvast.vulkan.Ffm.gl;
import static dev.supirvast.vulkan.Ffm.invoke;
import static dev.supirvast.vulkan.Ffm.invokeVoid;
import static dev.supirvast.vulkan.Ffm.sa;
import static dev.supirvast.vulkan.Ffm.si;
import static dev.supirvast.vulkan.Ffm.sl;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The Vulkan objects a compute context makes and the commands it records, over a {@link VulkanDevice}: the
 * Panama form of what the helper methods under {@link GpuContext} used to do through LWJGL, one for one.
 *
 * <p>Every command is resolved once, here, through the device's own {@code vkGetDeviceProcAddr}, so a context
 * built on a device that was made for something else — a window's, say — uses that device's commands and no
 * others. Handles of objects the device owns are {@code long}s, as they are in the specification; the three
 * dispatchable ones (device, queue, command buffer) are {@link MemorySegment}s.
 *
 * <p><b>Nothing here is synchronised, and a {@code VkQueue} must be.</b> A caller that shares the device with
 * something that draws submits from the same thread, or under a lock both agree on.
 *
 * <p>What Vulkan does not ask to be synchronised is safe from any thread: making and destroying buffers,
 * memory, shader modules, layouts and pipelines, each call over objects of its own. Every call here builds its
 * structs in an arena of its own and the command handles are fixed at construction, so this class adds no
 * shared state to that. A command pool, and every command buffer allocated from it, is another matter: Vulkan
 * requires those to be externally synchronised, and so does a queue.
 *
 * <p>Failures are {@link IllegalStateException}s naming the call and the {@code VkResult}, as they have
 * always been from this context: callers distinguish "this machine cannot" from a bug by that type.
 */
final class VkCompute {

    // VkStructureType
    private static final int STYPE_SUBMIT_INFO = 4;
    private static final int STYPE_MEMORY_ALLOCATE_INFO = 5;
    private static final int STYPE_MAPPED_MEMORY_RANGE = 6;
    private static final int STYPE_FENCE_CREATE_INFO = 8;
    private static final int STYPE_BUFFER_CREATE_INFO = 12;
    private static final int STYPE_SHADER_MODULE_CREATE_INFO = 16;
    private static final int STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
    private static final int STYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
    private static final int STYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
    private static final int STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
    private static final int STYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
    private static final int STYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
    private static final int STYPE_WRITE_DESCRIPTOR_SET = 35;
    private static final int STYPE_COMMAND_POOL_CREATE_INFO = 39;
    private static final int STYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
    private static final int STYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
    private static final int STYPE_MEMORY_BARRIER = 46;
    private static final int STYPE_SEMAPHORE_CREATE_INFO = 9;
    private static final int STYPE_QUERY_POOL_CREATE_INFO = 11;
    private static final int STYPE_SEMAPHORE_TYPE_CREATE_INFO = 1000207002;
    private static final int STYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003;
    private static final int STYPE_SEMAPHORE_WAIT_INFO = 1000207004;
    private static final int STYPE_SEMAPHORE_SIGNAL_INFO = 1000207005;
    private static final int STYPE_REQUIRED_SUBGROUP_SIZE = 1000225001;

    static final int BUFFER_USAGE_TRANSFER_SRC = 0x01;
    static final int BUFFER_USAGE_TRANSFER_DST = 0x02;
    static final int BUFFER_USAGE_STORAGE = 0x20;

    static final int MEMORY_DEVICE_LOCAL = 0x01;
    static final int MEMORY_HOST_VISIBLE = 0x02;
    static final int MEMORY_HOST_COHERENT = 0x04;
    static final int MEMORY_HOST_CACHED = 0x08;

    static final int COMMAND_BUFFER_ONE_TIME_SUBMIT = 0x01;
    static final int COMMAND_BUFFER_SIMULTANEOUS_USE = 0x04;

    private static final int DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    private static final int SHADER_STAGE_COMPUTE = 0x20;
    private static final int PIPELINE_BIND_POINT_COMPUTE = 1;
    private static final int PIPELINE_STAGE_COMPUTE_SHADER = 0x0800;
    private static final int PIPELINE_STAGE_TRANSFER = 0x1000;
    private static final int PIPELINE_STAGE_HOST = 0x4000;
    private static final int ACCESS_SHADER_READ = 0x0020;
    private static final int ACCESS_SHADER_WRITE = 0x0040;
    private static final int ACCESS_TRANSFER_READ = 0x0800;
    private static final int ACCESS_TRANSFER_WRITE = 0x1000;
    private static final int ACCESS_HOST_READ = 0x2000;
    private static final int PIPELINE_SHADER_STAGE_REQUIRE_FULL_SUBGROUPS = 0x02;
    private static final int COMMAND_POOL_RESET_COMMAND_BUFFER = 0x02;
    private static final int COMMAND_BUFFER_LEVEL_PRIMARY = 0;
    private static final int SHARING_MODE_EXCLUSIVE = 0;
    private static final int SHARING_MODE_CONCURRENT = 1;
    private static final int SEMAPHORE_TYPE_TIMELINE = 1;
    private static final int PIPELINE_STAGE_ALL_COMMANDS = 0x10000;
    private static final int VK_TIMEOUT = 2;
    private static final int QUERY_TYPE_TIMESTAMP = 2;
    private static final int QUERY_RESULT_64 = 0x1;
    private static final int VK_NOT_READY = 1;
    static final int PIPELINE_STAGE_TOP_OF_PIPE = 0x1;
    static final int PIPELINE_STAGE_BOTTOM_OF_PIPE = 0x2000;
    private static final long WHOLE_SIZE = ~0L;
    private static final int VK_SUCCESS = 0;
    private static final int RESULT_BYTES = Integer.BYTES;

    private static final GroupLayout BUFFER_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("size"),
            JAVA_INT.withName("usage"), JAVA_INT.withName("sharingMode"),
            JAVA_INT.withName("queueFamilyIndexCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pQueueFamilyIndices")).withName("VkBufferCreateInfo");

    private static final GroupLayout MEMORY_REQUIREMENTS = MemoryLayout.structLayout(
            JAVA_LONG.withName("size"), JAVA_LONG.withName("alignment"),
            JAVA_INT.withName("memoryTypeBits"), MemoryLayout.paddingLayout(4)).withName("VkMemoryRequirements");

    private static final GroupLayout MEMORY_ALLOCATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("allocationSize"), JAVA_INT.withName("memoryTypeIndex"), MemoryLayout.paddingLayout(4)
    ).withName("VkMemoryAllocateInfo");

    private static final GroupLayout MAPPED_MEMORY_RANGE = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("memory"), JAVA_LONG.withName("offset"), JAVA_LONG.withName("size")
    ).withName("VkMappedMemoryRange");

    private static final GroupLayout SHADER_MODULE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("codeSize"),
            ADDRESS.withName("pCode")).withName("VkShaderModuleCreateInfo");

    private static final GroupLayout DESCRIPTOR_SET_LAYOUT_BINDING = MemoryLayout.structLayout(
            JAVA_INT.withName("binding"), JAVA_INT.withName("descriptorType"), JAVA_INT.withName("descriptorCount"),
            JAVA_INT.withName("stageFlags"), ADDRESS.withName("pImmutableSamplers")
    ).withName("VkDescriptorSetLayoutBinding");

    private static final GroupLayout DESCRIPTOR_SET_LAYOUT_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("bindingCount"), ADDRESS.withName("pBindings")
    ).withName("VkDescriptorSetLayoutCreateInfo");

    private static final GroupLayout PUSH_CONSTANT_RANGE = MemoryLayout.structLayout(
            JAVA_INT.withName("stageFlags"), JAVA_INT.withName("offset"), JAVA_INT.withName("size")
    ).withName("VkPushConstantRange");

    private static final GroupLayout PIPELINE_LAYOUT_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("setLayoutCount"), ADDRESS.withName("pSetLayouts"),
            JAVA_INT.withName("pushConstantRangeCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pPushConstantRanges")).withName("VkPipelineLayoutCreateInfo");

    private static final GroupLayout PIPELINE_SHADER_STAGE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("stage"), JAVA_LONG.withName("module"),
            ADDRESS.withName("pName"), ADDRESS.withName("pSpecializationInfo")
    ).withName("VkPipelineShaderStageCreateInfo");

    private static final GroupLayout REQUIRED_SUBGROUP_SIZE = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("requiredSubgroupSize"), MemoryLayout.paddingLayout(4)
    ).withName("VkPipelineShaderStageRequiredSubgroupSizeCreateInfo");

    private static final GroupLayout COMPUTE_PIPELINE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4),
            PIPELINE_SHADER_STAGE_CREATE_INFO.withName("stage"),
            JAVA_LONG.withName("layout"), JAVA_LONG.withName("basePipelineHandle"),
            JAVA_INT.withName("basePipelineIndex"), MemoryLayout.paddingLayout(4)
    ).withName("VkComputePipelineCreateInfo");

    private static final GroupLayout DESCRIPTOR_POOL_SIZE = MemoryLayout.structLayout(
            JAVA_INT.withName("type"), JAVA_INT.withName("descriptorCount")).withName("VkDescriptorPoolSize");

    private static final GroupLayout DESCRIPTOR_POOL_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("maxSets"), JAVA_INT.withName("poolSizeCount"),
            MemoryLayout.paddingLayout(4), ADDRESS.withName("pPoolSizes")).withName("VkDescriptorPoolCreateInfo");

    private static final GroupLayout DESCRIPTOR_SET_ALLOCATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("descriptorPool"), JAVA_INT.withName("descriptorSetCount"),
            MemoryLayout.paddingLayout(4), ADDRESS.withName("pSetLayouts")).withName("VkDescriptorSetAllocateInfo");

    private static final GroupLayout DESCRIPTOR_BUFFER_INFO = MemoryLayout.structLayout(
            JAVA_LONG.withName("buffer"), JAVA_LONG.withName("offset"), JAVA_LONG.withName("range")
    ).withName("VkDescriptorBufferInfo");

    private static final GroupLayout WRITE_DESCRIPTOR_SET = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("dstSet"), JAVA_INT.withName("dstBinding"), JAVA_INT.withName("dstArrayElement"),
            JAVA_INT.withName("descriptorCount"), JAVA_INT.withName("descriptorType"), ADDRESS.withName("pImageInfo"),
            ADDRESS.withName("pBufferInfo"), ADDRESS.withName("pTexelBufferView")).withName("VkWriteDescriptorSet");

    private static final GroupLayout COMMAND_POOL_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("queueFamilyIndex")).withName("VkCommandPoolCreateInfo");

    private static final GroupLayout COMMAND_BUFFER_ALLOCATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("commandPool"), JAVA_INT.withName("level"), JAVA_INT.withName("commandBufferCount")
    ).withName("VkCommandBufferAllocateInfo");

    private static final GroupLayout COMMAND_BUFFER_BEGIN_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pInheritanceInfo")
    ).withName("VkCommandBufferBeginInfo");

    private static final GroupLayout MEMORY_BARRIER = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("srcAccessMask"), JAVA_INT.withName("dstAccessMask")).withName("VkMemoryBarrier");

    private static final GroupLayout BUFFER_COPY = MemoryLayout.structLayout(
            JAVA_LONG.withName("srcOffset"), JAVA_LONG.withName("dstOffset"), JAVA_LONG.withName("size")
    ).withName("VkBufferCopy");

    private static final GroupLayout FENCE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4)).withName("VkFenceCreateInfo");

    private static final GroupLayout SUBMIT_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("waitSemaphoreCount"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pWaitSemaphores"),
            ADDRESS.withName("pWaitDstStageMask"), JAVA_INT.withName("commandBufferCount"),
            MemoryLayout.paddingLayout(4), ADDRESS.withName("pCommandBuffers"),
            JAVA_INT.withName("signalSemaphoreCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pSignalSemaphores")).withName("VkSubmitInfo");

    private static final GroupLayout SEMAPHORE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4)).withName("VkSemaphoreCreateInfo");

    private static final GroupLayout SEMAPHORE_TYPE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("semaphoreType"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("initialValue")
    ).withName("VkSemaphoreTypeCreateInfo");

    private static final GroupLayout SEMAPHORE_WAIT_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("semaphoreCount"), ADDRESS.withName("pSemaphores"),
            ADDRESS.withName("pValues")).withName("VkSemaphoreWaitInfo");

    private static final GroupLayout SEMAPHORE_SIGNAL_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("semaphore"), JAVA_LONG.withName("value")).withName("VkSemaphoreSignalInfo");

    private static final GroupLayout TIMELINE_SEMAPHORE_SUBMIT_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("waitSemaphoreValueCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pWaitSemaphoreValues"), JAVA_INT.withName("signalSemaphoreValueCount"),
            MemoryLayout.paddingLayout(4), ADDRESS.withName("pSignalSemaphoreValues")
    ).withName("VkTimelineSemaphoreSubmitInfo");

    private static final GroupLayout QUERY_POOL_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("queryType"), JAVA_INT.withName("queryCount"),
            JAVA_INT.withName("pipelineStatistics")).withName("VkQueryPoolCreateInfo");

    private static final FunctionDescriptor CREATE =
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS);
    private static final FunctionDescriptor DESTROY = FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS);

    private final VulkanDevice device;
    private final MemorySegment dev;

    private final MethodHandle vkCreateBuffer;
    private final MethodHandle vkDestroyBuffer;
    private final MethodHandle vkGetBufferMemoryRequirements;
    private final MethodHandle vkAllocateMemory;
    private final MethodHandle vkFreeMemory;
    private final MethodHandle vkBindBufferMemory;
    private final MethodHandle vkMapMemory;
    private final MethodHandle vkUnmapMemory;
    private final MethodHandle vkInvalidateMappedMemoryRanges;
    private final MethodHandle vkCreateShaderModule;
    private final MethodHandle vkDestroyShaderModule;
    private final MethodHandle vkCreateDescriptorSetLayout;
    private final MethodHandle vkDestroyDescriptorSetLayout;
    private final MethodHandle vkCreatePipelineLayout;
    private final MethodHandle vkDestroyPipelineLayout;
    private final MethodHandle vkCreateComputePipelines;
    private final MethodHandle vkDestroyPipeline;
    private final MethodHandle vkCreateDescriptorPool;
    private final MethodHandle vkDestroyDescriptorPool;
    private final MethodHandle vkAllocateDescriptorSets;
    private final MethodHandle vkUpdateDescriptorSets;
    private final MethodHandle vkCreateCommandPool;
    private final MethodHandle vkDestroyCommandPool;
    private final MethodHandle vkAllocateCommandBuffers;
    private final MethodHandle vkFreeCommandBuffers;
    private final MethodHandle vkBeginCommandBuffer;
    private final MethodHandle vkEndCommandBuffer;
    private final MethodHandle vkCmdBindPipeline;
    private final MethodHandle vkCmdBindDescriptorSets;
    private final MethodHandle vkCmdPushConstants;
    private final MethodHandle vkCmdDispatch;
    private final MethodHandle vkCmdPipelineBarrier;
    private final MethodHandle vkCmdCopyBuffer;
    private final MethodHandle vkCmdFillBuffer;
    private final MethodHandle vkCreateFence;
    private final MethodHandle vkDestroyFence;
    private final MethodHandle vkWaitForFences;
    private final MethodHandle vkGetFenceStatus;
    private final MethodHandle vkQueueSubmit;
    private final MethodHandle vkCreateSemaphore;
    private final MethodHandle vkDestroySemaphore;
    private final MethodHandle vkGetSemaphoreCounterValue;
    private final MethodHandle vkWaitSemaphores;
    private final MethodHandle vkSignalSemaphore;
    private final MethodHandle vkCreateQueryPool;
    private final MethodHandle vkDestroyQueryPool;
    private final MethodHandle vkCmdResetQueryPool;
    private final MethodHandle vkCmdWriteTimestamp;
    private final MethodHandle vkGetQueryPoolResults;
    /** The device's queue families, when there is more than one, for buffers every one of them may use; else null. */
    private final int[] sharingFamilies;

    VkCompute(VulkanDevice device) {
        this.device = device;
        this.dev = device.handle();
        vkCreateBuffer = device.command("vkCreateBuffer", CREATE);
        vkDestroyBuffer = device.command("vkDestroyBuffer", DESTROY);
        vkGetBufferMemoryRequirements = device.command("vkGetBufferMemoryRequirements",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS));
        vkAllocateMemory = device.command("vkAllocateMemory", CREATE);
        vkFreeMemory = device.command("vkFreeMemory", DESTROY);
        vkBindBufferMemory = device.command("vkBindBufferMemory",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG));
        vkMapMemory = device.command("vkMapMemory",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS));
        vkUnmapMemory = device.command("vkUnmapMemory", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
        vkInvalidateMappedMemoryRanges = device.command("vkInvalidateMappedMemoryRanges",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
        vkCreateShaderModule = device.command("vkCreateShaderModule", CREATE);
        vkDestroyShaderModule = device.command("vkDestroyShaderModule", DESTROY);
        vkCreateDescriptorSetLayout = device.command("vkCreateDescriptorSetLayout", CREATE);
        vkDestroyDescriptorSetLayout = device.command("vkDestroyDescriptorSetLayout", DESTROY);
        vkCreatePipelineLayout = device.command("vkCreatePipelineLayout", CREATE);
        vkDestroyPipelineLayout = device.command("vkDestroyPipelineLayout", DESTROY);
        vkCreateComputePipelines = device.command("vkCreateComputePipelines",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        vkDestroyPipeline = device.command("vkDestroyPipeline", DESTROY);
        vkCreateDescriptorPool = device.command("vkCreateDescriptorPool", CREATE);
        vkDestroyDescriptorPool = device.command("vkDestroyDescriptorPool", DESTROY);
        vkAllocateDescriptorSets = device.command("vkAllocateDescriptorSets",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        vkUpdateDescriptorSets = device.command("vkUpdateDescriptorSets",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
        vkCreateCommandPool = device.command("vkCreateCommandPool", CREATE);
        vkDestroyCommandPool = device.command("vkDestroyCommandPool", DESTROY);
        vkAllocateCommandBuffers = device.command("vkAllocateCommandBuffers",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        vkFreeCommandBuffers = device.command("vkFreeCommandBuffers",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
        vkBeginCommandBuffer = device.command("vkBeginCommandBuffer",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        vkEndCommandBuffer = device.command("vkEndCommandBuffer", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        vkCmdBindPipeline = device.command("vkCmdBindPipeline",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_LONG));
        vkCmdBindDescriptorSets = device.command("vkCmdBindDescriptorSets",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT,
                        ADDRESS));
        vkCmdPushConstants = device.command("vkCmdPushConstants",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS));
        vkCmdDispatch = device.command("vkCmdDispatch",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
        vkCmdPipelineBarrier = device.command("vkCmdPipelineBarrier",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT,
                        ADDRESS, JAVA_INT, ADDRESS));
        vkCmdCopyBuffer = device.command("vkCmdCopyBuffer",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS));
        vkCmdFillBuffer = device.command("vkCmdFillBuffer",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));
        vkCreateFence = device.command("vkCreateFence", CREATE);
        vkDestroyFence = device.command("vkDestroyFence", DESTROY);
        vkWaitForFences = device.command("vkWaitForFences",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG));
        vkGetFenceStatus = device.command("vkGetFenceStatus",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
        vkQueueSubmit = device.command("vkQueueSubmit",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG));
        // Core since 1.2, so always resolvable on the 1.3 devices this runs on; usable only where the device was
        // made with the timeline feature, which the context checks before it makes one.
        vkCreateSemaphore = device.command("vkCreateSemaphore", CREATE);
        vkDestroySemaphore = device.command("vkDestroySemaphore", DESTROY);
        vkGetSemaphoreCounterValue = device.command("vkGetSemaphoreCounterValue",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
        vkWaitSemaphores = device.command("vkWaitSemaphores",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
        vkSignalSemaphore = device.command("vkSignalSemaphore", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        vkCreateQueryPool = device.command("vkCreateQueryPool", CREATE);
        vkDestroyQueryPool = device.command("vkDestroyQueryPool", DESTROY);
        vkCmdResetQueryPool = device.command("vkCmdResetQueryPool",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT));
        vkCmdWriteTimestamp = device.command("vkCmdWriteTimestamp",
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_LONG, JAVA_INT));
        vkGetQueryPoolResults = device.command("vkGetQueryPoolResults", FunctionDescriptor.of(JAVA_INT, ADDRESS,
                JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT));
        java.util.List<Integer> families = device.queueFamilies();
        sharingFamilies = families.size() > 1 ? families.stream().mapToInt(Integer::intValue).toArray() : null;
    }

    // --- buffers and memory ----------------------------------------------------------------------------

    /** Bound device memory, and whether host writes and reads of it need no flush or invalidate. */
    record Allocation(long memory, boolean coherent) {
    }

    /**
     * A buffer of {@code sizeBytes}. On a device with queues of several families, every one of them may use it
     * (concurrent sharing), so a buffer one queue computes into can be read on another without an ownership
     * transfer; the waits between their work are the caller's, by semaphore.
     */
    long createBuffer(long sizeBytes, int usage) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(BUFFER_CREATE_INFO);
            si(info, BUFFER_CREATE_INFO, "sType", STYPE_BUFFER_CREATE_INFO);
            sl(info, BUFFER_CREATE_INFO, "size", sizeBytes);
            si(info, BUFFER_CREATE_INFO, "usage", usage);
            if (sharingFamilies == null) {
                si(info, BUFFER_CREATE_INFO, "sharingMode", SHARING_MODE_EXCLUSIVE);
            } else {
                si(info, BUFFER_CREATE_INFO, "sharingMode", SHARING_MODE_CONCURRENT);
                si(info, BUFFER_CREATE_INFO, "queueFamilyIndexCount", sharingFamilies.length);
                sa(info, BUFFER_CREATE_INFO, "pQueueFamilyIndices", a.allocateFrom(JAVA_INT, sharingFamilies));
            }
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateBuffer, dev, info, MemorySegment.NULL, out), "vkCreateBuffer");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyBuffer(long buffer) {
        invokeVoid(vkDestroyBuffer, dev, buffer, MemorySegment.NULL);
    }

    void freeMemory(long memory) {
        invokeVoid(vkFreeMemory, dev, memory, MemorySegment.NULL);
    }

    /** Host-visible, host-coherent memory bound to {@code buffer} — the per-dispatch path's staging-free road. */
    long allocateHostVisible(long buffer) {
        return allocate(buffer, 0, MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT).memory();
    }

    /**
     * Allocates and binds memory for {@code buffer} from a type that has every {@code required} property,
     * preferring one that also has every {@code preferred} property.
     */
    Allocation allocate(long buffer, int preferred, int required) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment requirements = a.allocate(MEMORY_REQUIREMENTS);
            invokeVoid(vkGetBufferMemoryRequirements, dev, buffer, requirements);
            int typeBits = gi(requirements, MEMORY_REQUIREMENTS, "memoryTypeBits");
            long size = gl(requirements, MEMORY_REQUIREMENTS, "size");

            int type = device.tryFindMemoryType(typeBits, required | preferred);
            if (type < 0) {
                type = device.tryFindMemoryType(typeBits, required);
            }
            if (type < 0) {
                throw new IllegalStateException("no memory type with properties 0x" + Integer.toHexString(required)
                        + " for the buffer");
            }

            MemorySegment info = a.allocate(MEMORY_ALLOCATE_INFO);
            si(info, MEMORY_ALLOCATE_INFO, "sType", STYPE_MEMORY_ALLOCATE_INFO);
            sl(info, MEMORY_ALLOCATE_INFO, "allocationSize", size);
            si(info, MEMORY_ALLOCATE_INFO, "memoryTypeIndex", type);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkAllocateMemory, dev, info, MemorySegment.NULL, out), "vkAllocateMemory");
            long memory = out.get(JAVA_LONG, 0);
            check(invoke(vkBindBufferMemory, dev, buffer, memory, 0L), "vkBindBufferMemory");
            return new Allocation(memory, device.memoryTypeIsHostCoherent(type));
        }
    }

    void writeInts(long memory, int[] data) {
        long size = Math.max(RESULT_BYTES, (long) data.length * Integer.BYTES);
        MemorySegment mapped = map(memory, size);
        MemorySegment.copy(data, 0, mapped, JAVA_INT, 0, data.length);
        unmap(memory);
    }

    int[] readInts(long memory, int length) {
        long size = Math.max(RESULT_BYTES, (long) length * Integer.BYTES);
        MemorySegment mapped = map(memory, size);
        int[] out = new int[length];
        MemorySegment.copy(mapped, JAVA_INT, 0, out, 0, length);
        unmap(memory);
        return out;
    }

    /** Make what the device wrote to non-coherent memory visible to the host's next read of it. */
    void invalidate(long memory) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment range = a.allocate(MAPPED_MEMORY_RANGE);
            si(range, MAPPED_MEMORY_RANGE, "sType", STYPE_MAPPED_MEMORY_RANGE);
            sl(range, MAPPED_MEMORY_RANGE, "memory", memory);
            sl(range, MAPPED_MEMORY_RANGE, "offset", 0L);
            sl(range, MAPPED_MEMORY_RANGE, "size", WHOLE_SIZE);
            check(invoke(vkInvalidateMappedMemoryRanges, dev, 1, range), "vkInvalidateMappedMemoryRanges");
        }
    }

    /** {@code size} bytes of {@code memory}, mapped until it is unmapped or freed. */
    MemorySegment map(long memory, long size) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pData = a.allocate(ADDRESS);
            check(invoke(vkMapMemory, dev, memory, 0L, size, 0, pData), "vkMapMemory");
            return pData.get(ADDRESS, 0).reinterpret(size);
        }
    }

    private void unmap(long memory) {
        invokeVoid(vkUnmapMemory, dev, memory);
    }

    // --- pipelines -------------------------------------------------------------------------------------

    long createShaderModule(byte[] spirv) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment code = a.allocate(spirv.length, 4);
            MemorySegment.copy(spirv, 0, code, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, spirv.length);
            MemorySegment info = a.allocate(SHADER_MODULE_CREATE_INFO);
            si(info, SHADER_MODULE_CREATE_INFO, "sType", STYPE_SHADER_MODULE_CREATE_INFO);
            sl(info, SHADER_MODULE_CREATE_INFO, "codeSize", spirv.length);
            sa(info, SHADER_MODULE_CREATE_INFO, "pCode", code);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateShaderModule, dev, info, MemorySegment.NULL, out), "vkCreateShaderModule");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyShaderModule(long module) {
        invokeVoid(vkDestroyShaderModule, dev, module, MemorySegment.NULL);
    }

    /** One storage buffer per binding {@code 0..bindingCount-1}, all visible to the compute stage. */
    long createSetLayout(int bindingCount) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment bindings = a.allocate(DESCRIPTOR_SET_LAYOUT_BINDING, bindingCount);
            long stride = DESCRIPTOR_SET_LAYOUT_BINDING.byteSize();
            for (int i = 0; i < bindingCount; i++) {
                MemorySegment binding = bindings.asSlice(i * stride, stride);
                si(binding, DESCRIPTOR_SET_LAYOUT_BINDING, "binding", i);
                si(binding, DESCRIPTOR_SET_LAYOUT_BINDING, "descriptorType", DESCRIPTOR_TYPE_STORAGE_BUFFER);
                si(binding, DESCRIPTOR_SET_LAYOUT_BINDING, "descriptorCount", 1);
                si(binding, DESCRIPTOR_SET_LAYOUT_BINDING, "stageFlags", SHADER_STAGE_COMPUTE);
            }
            MemorySegment info = a.allocate(DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
            si(info, DESCRIPTOR_SET_LAYOUT_CREATE_INFO, "sType", STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
            si(info, DESCRIPTOR_SET_LAYOUT_CREATE_INFO, "bindingCount", bindingCount);
            sa(info, DESCRIPTOR_SET_LAYOUT_CREATE_INFO, "pBindings", bindings);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateDescriptorSetLayout, dev, info, MemorySegment.NULL, out),
                    "vkCreateDescriptorSetLayout");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroySetLayout(long layout) {
        invokeVoid(vkDestroyDescriptorSetLayout, dev, layout, MemorySegment.NULL);
    }

    /** Set 0, and the 4-byte invocation count at push-constant offset 0 that every dispatch sets. */
    long createPipelineLayout(long setLayout) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment sets = a.allocate(JAVA_LONG);
            sets.set(JAVA_LONG, 0, setLayout);
            MemorySegment range = a.allocate(PUSH_CONSTANT_RANGE);
            si(range, PUSH_CONSTANT_RANGE, "stageFlags", SHADER_STAGE_COMPUTE);
            si(range, PUSH_CONSTANT_RANGE, "offset", 0);
            si(range, PUSH_CONSTANT_RANGE, "size", Integer.BYTES);
            MemorySegment info = a.allocate(PIPELINE_LAYOUT_CREATE_INFO);
            si(info, PIPELINE_LAYOUT_CREATE_INFO, "sType", STYPE_PIPELINE_LAYOUT_CREATE_INFO);
            si(info, PIPELINE_LAYOUT_CREATE_INFO, "setLayoutCount", 1);
            sa(info, PIPELINE_LAYOUT_CREATE_INFO, "pSetLayouts", sets);
            si(info, PIPELINE_LAYOUT_CREATE_INFO, "pushConstantRangeCount", 1);
            sa(info, PIPELINE_LAYOUT_CREATE_INFO, "pPushConstantRanges", range);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreatePipelineLayout, dev, info, MemorySegment.NULL, out), "vkCreatePipelineLayout");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyPipelineLayout(long layout) {
        invokeVoid(vkDestroyPipelineLayout, dev, layout, MemorySegment.NULL);
    }

    /** @param subgroupSize full subgroups of exactly this many lanes, or 0 to leave it to the device */
    long createComputePipeline(long layout, long shaderModule, String entryPoint, int subgroupSize) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(COMPUTE_PIPELINE_CREATE_INFO);
            si(info, COMPUTE_PIPELINE_CREATE_INFO, "sType", STYPE_COMPUTE_PIPELINE_CREATE_INFO);
            sl(info, COMPUTE_PIPELINE_CREATE_INFO, "layout", layout);
            sl(info, COMPUTE_PIPELINE_CREATE_INFO, "basePipelineHandle", 0L);
            si(info, COMPUTE_PIPELINE_CREATE_INFO, "basePipelineIndex", -1);

            long stageOffset = Ffm.off(COMPUTE_PIPELINE_CREATE_INFO, "stage");
            MemorySegment stage = info.asSlice(stageOffset, PIPELINE_SHADER_STAGE_CREATE_INFO.byteSize());
            si(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "sType", STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
            si(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "stage", SHADER_STAGE_COMPUTE);
            sl(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "module", shaderModule);
            sa(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "pName", a.allocateFrom(entryPoint));
            if (subgroupSize != 0) {
                // Both halves matter: the size fixes which invocations share a subgroup, and full subgroups mean
                // none of them is missing a lane — without which a reduction would quietly cover fewer values.
                MemorySegment required = a.allocate(REQUIRED_SUBGROUP_SIZE);
                si(required, REQUIRED_SUBGROUP_SIZE, "sType", STYPE_REQUIRED_SUBGROUP_SIZE);
                si(required, REQUIRED_SUBGROUP_SIZE, "requiredSubgroupSize", subgroupSize);
                si(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "flags", PIPELINE_SHADER_STAGE_REQUIRE_FULL_SUBGROUPS);
                sa(stage, PIPELINE_SHADER_STAGE_CREATE_INFO, "pNext", required);
            }
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateComputePipelines, dev, 0L, 1, info, MemorySegment.NULL, out),
                    "vkCreateComputePipelines");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyPipeline(long pipeline) {
        invokeVoid(vkDestroyPipeline, dev, pipeline, MemorySegment.NULL);
    }

    // --- descriptors -----------------------------------------------------------------------------------

    long createDescriptorPool(int descriptorCount, int sets) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment size = a.allocate(DESCRIPTOR_POOL_SIZE);
            si(size, DESCRIPTOR_POOL_SIZE, "type", DESCRIPTOR_TYPE_STORAGE_BUFFER);
            si(size, DESCRIPTOR_POOL_SIZE, "descriptorCount", Math.max(1, descriptorCount));
            MemorySegment info = a.allocate(DESCRIPTOR_POOL_CREATE_INFO);
            si(info, DESCRIPTOR_POOL_CREATE_INFO, "sType", STYPE_DESCRIPTOR_POOL_CREATE_INFO);
            si(info, DESCRIPTOR_POOL_CREATE_INFO, "maxSets", sets);
            si(info, DESCRIPTOR_POOL_CREATE_INFO, "poolSizeCount", 1);
            sa(info, DESCRIPTOR_POOL_CREATE_INFO, "pPoolSizes", size);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateDescriptorPool, dev, info, MemorySegment.NULL, out), "vkCreateDescriptorPool");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyDescriptorPool(long pool) {
        invokeVoid(vkDestroyDescriptorPool, dev, pool, MemorySegment.NULL);
    }

    long allocateDescriptorSet(long pool, long setLayout) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment layouts = a.allocate(JAVA_LONG);
            layouts.set(JAVA_LONG, 0, setLayout);
            MemorySegment info = a.allocate(DESCRIPTOR_SET_ALLOCATE_INFO);
            si(info, DESCRIPTOR_SET_ALLOCATE_INFO, "sType", STYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
            sl(info, DESCRIPTOR_SET_ALLOCATE_INFO, "descriptorPool", pool);
            si(info, DESCRIPTOR_SET_ALLOCATE_INFO, "descriptorSetCount", 1);
            sa(info, DESCRIPTOR_SET_ALLOCATE_INFO, "pSetLayouts", layouts);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkAllocateDescriptorSets, dev, info, out), "vkAllocateDescriptorSets");
            return out.get(JAVA_LONG, 0);
        }
    }

    /** Point binding {@code i} of {@code descriptorSet} at {@code buffers[i]}, whole. */
    void bindBuffers(long descriptorSet, long[] buffers) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment writes = a.allocate(WRITE_DESCRIPTOR_SET, buffers.length);
            MemorySegment infos = a.allocate(DESCRIPTOR_BUFFER_INFO, buffers.length);
            long writeStride = WRITE_DESCRIPTOR_SET.byteSize();
            long infoStride = DESCRIPTOR_BUFFER_INFO.byteSize();
            for (int i = 0; i < buffers.length; i++) {
                MemorySegment info = infos.asSlice(i * infoStride, infoStride);
                sl(info, DESCRIPTOR_BUFFER_INFO, "buffer", buffers[i]);
                sl(info, DESCRIPTOR_BUFFER_INFO, "offset", 0L);
                sl(info, DESCRIPTOR_BUFFER_INFO, "range", WHOLE_SIZE);
                MemorySegment write = writes.asSlice(i * writeStride, writeStride);
                si(write, WRITE_DESCRIPTOR_SET, "sType", STYPE_WRITE_DESCRIPTOR_SET);
                sl(write, WRITE_DESCRIPTOR_SET, "dstSet", descriptorSet);
                si(write, WRITE_DESCRIPTOR_SET, "dstBinding", i);
                si(write, WRITE_DESCRIPTOR_SET, "descriptorCount", 1);
                si(write, WRITE_DESCRIPTOR_SET, "descriptorType", DESCRIPTOR_TYPE_STORAGE_BUFFER);
                sa(write, WRITE_DESCRIPTOR_SET, "pBufferInfo", info);
            }
            invokeVoid(vkUpdateDescriptorSets, dev, buffers.length, writes, 0, MemorySegment.NULL);
        }
    }

    // --- commands --------------------------------------------------------------------------------------

    long createCommandPool(int queueFamily) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(COMMAND_POOL_CREATE_INFO);
            si(info, COMMAND_POOL_CREATE_INFO, "sType", STYPE_COMMAND_POOL_CREATE_INFO);
            si(info, COMMAND_POOL_CREATE_INFO, "flags", COMMAND_POOL_RESET_COMMAND_BUFFER);
            si(info, COMMAND_POOL_CREATE_INFO, "queueFamilyIndex", queueFamily);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateCommandPool, dev, info, MemorySegment.NULL, out), "vkCreateCommandPool");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyCommandPool(long pool) {
        invokeVoid(vkDestroyCommandPool, dev, pool, MemorySegment.NULL);
    }

    /** A primary command buffer, begun with {@code usage} ({@code COMMAND_BUFFER_*} flags). */
    MemorySegment beginCommandBuffer(long pool, int usage) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment allocate = a.allocate(COMMAND_BUFFER_ALLOCATE_INFO);
            si(allocate, COMMAND_BUFFER_ALLOCATE_INFO, "sType", STYPE_COMMAND_BUFFER_ALLOCATE_INFO);
            sl(allocate, COMMAND_BUFFER_ALLOCATE_INFO, "commandPool", pool);
            si(allocate, COMMAND_BUFFER_ALLOCATE_INFO, "level", COMMAND_BUFFER_LEVEL_PRIMARY);
            si(allocate, COMMAND_BUFFER_ALLOCATE_INFO, "commandBufferCount", 1);
            MemorySegment pCmd = a.allocate(ADDRESS);
            check(invoke(vkAllocateCommandBuffers, dev, allocate, pCmd), "vkAllocateCommandBuffers");
            MemorySegment cmd = pCmd.get(ADDRESS, 0);

            MemorySegment begin = a.allocate(COMMAND_BUFFER_BEGIN_INFO);
            si(begin, COMMAND_BUFFER_BEGIN_INFO, "sType", STYPE_COMMAND_BUFFER_BEGIN_INFO);
            si(begin, COMMAND_BUFFER_BEGIN_INFO, "flags", usage);
            check(invoke(vkBeginCommandBuffer, cmd, begin), "vkBeginCommandBuffer");
            return cmd;
        }
    }

    void endCommandBuffer(MemorySegment cmd) {
        check(invoke(vkEndCommandBuffer, cmd), "vkEndCommandBuffer");
    }

    void freeCommandBuffer(long pool, MemorySegment cmd) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pCmd = a.allocate(ADDRESS);
            pCmd.set(ADDRESS, 0, cmd);
            invokeVoid(vkFreeCommandBuffers, dev, pool, 1, pCmd);
        }
    }

    /** Bind a pipeline, its one descriptor set and the invocation count, then dispatch {@code groups} workgroups. */
    void recordDispatch(MemorySegment cmd, long pipeline, long pipelineLayout, long descriptorSet,
                        int invocations, int groups) {
        try (Arena a = Arena.ofConfined()) {
            invokeVoid(vkCmdBindPipeline, cmd, PIPELINE_BIND_POINT_COMPUTE, pipeline);
            MemorySegment sets = a.allocate(JAVA_LONG);
            sets.set(JAVA_LONG, 0, descriptorSet);
            invokeVoid(vkCmdBindDescriptorSets, cmd, PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, sets, 0,
                    MemorySegment.NULL);
            MemorySegment count = a.allocate(JAVA_INT);
            count.set(JAVA_INT, 0, invocations);
            invokeVoid(vkCmdPushConstants, cmd, pipelineLayout, SHADER_STAGE_COMPUTE, 0, Integer.BYTES, count);
            invokeVoid(vkCmdDispatch, cmd, groups, 1, 1);
        }
    }

    /**
     * Every earlier shader or transfer write, made visible to every later shader or transfer access. Coarse on
     * purpose: one barrier per command buffer is negligible against a dispatch, and a finer one would have to
     * know which buffers each dispatch touches — which is the kind of bookkeeping that is wrong once.
     */
    void residentBarrier(MemorySegment cmd) {
        int stages = PIPELINE_STAGE_COMPUTE_SHADER | PIPELINE_STAGE_TRANSFER;
        memoryBarrier(cmd, stages, stages, ACCESS_SHADER_WRITE | ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ | ACCESS_SHADER_WRITE | ACCESS_TRANSFER_READ | ACCESS_TRANSFER_WRITE);
    }

    /** A shader's write made visible to the host's read of it, for memory the host reads where it is. */
    void shaderToHostBarrier(MemorySegment cmd) {
        memoryBarrier(cmd, PIPELINE_STAGE_COMPUTE_SHADER, PIPELINE_STAGE_HOST, ACCESS_SHADER_WRITE, ACCESS_HOST_READ);
    }

    /** A transfer write made visible to the host's read of it. */
    void transferToHostBarrier(MemorySegment cmd) {
        memoryBarrier(cmd, PIPELINE_STAGE_TRANSFER, PIPELINE_STAGE_HOST, ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ);
    }

    private void memoryBarrier(MemorySegment cmd, int srcStages, int dstStages, int srcAccess, int dstAccess) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment barrier = a.allocate(MEMORY_BARRIER);
            si(barrier, MEMORY_BARRIER, "sType", STYPE_MEMORY_BARRIER);
            si(barrier, MEMORY_BARRIER, "srcAccessMask", srcAccess);
            si(barrier, MEMORY_BARRIER, "dstAccessMask", dstAccess);
            invokeVoid(vkCmdPipelineBarrier, cmd, srcStages, dstStages, 0, 1, barrier, 0, MemorySegment.NULL, 0,
                    MemorySegment.NULL);
        }
    }

    /** Copy the first {@code sizeBytes} of {@code source} over the start of {@code target}. */
    void recordCopy(MemorySegment cmd, long source, long target, long sizeBytes) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment region = a.allocate(BUFFER_COPY);
            sl(region, BUFFER_COPY, "size", sizeBytes);
            invokeVoid(vkCmdCopyBuffer, cmd, source, target, 1, region);
        }
    }

    /** Every word of {@code target} set to {@code word}, on the device: nothing is staged or copied from the host. */
    void recordFill(MemorySegment cmd, long target, int word) {
        invokeVoid(vkCmdFillBuffer, cmd, target, 0L, WHOLE_SIZE, word);
    }

    // --- fences and queues -----------------------------------------------------------------------------

    long createFence() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(FENCE_CREATE_INFO);
            si(info, FENCE_CREATE_INFO, "sType", STYPE_FENCE_CREATE_INFO);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateFence, dev, info, MemorySegment.NULL, out), "vkCreateFence");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyFence(long fence) {
        invokeVoid(vkDestroyFence, dev, fence, MemorySegment.NULL);
    }

    void waitFence(long fence) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment fences = a.allocate(JAVA_LONG);
            fences.set(JAVA_LONG, 0, fence);
            check(invoke(vkWaitForFences, dev, 1, fences, 1, Long.MAX_VALUE), "vkWaitForFences");
        }
    }

    boolean fenceSignaled(long fence) {
        return invoke(vkGetFenceStatus, dev, fence) == VK_SUCCESS;
    }

    void submit(MemorySegment queue, MemorySegment cmd, long fence) {
        submit(queue, cmd, fence, new long[0], new long[0], new long[0], new long[0]);
    }

    /**
     * {@code cmd} submitted to {@code queue}, signalling {@code fence}, after every timeline semaphore of
     * {@code waits} reaches its value of {@code waitValues}, and setting each of {@code signals} to its value of
     * {@code signalValues} once it has finished. A wait holds back every command of the submission.
     */
    void submit(MemorySegment queue, MemorySegment cmd, long fence, long[] waits, long[] waitValues, long[] signals,
                long[] signalValues) {
        submit(queue, new MemorySegment[] {cmd}, fence, waits, waitValues, signals, signalValues);
    }

    /** As above, for several command buffers run in order as one submission. */
    void submit(MemorySegment queue, MemorySegment[] commandBuffers, long fence, long[] waits, long[] waitValues,
                long[] signals, long[] signalValues) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment cmds = a.allocate(ADDRESS, commandBuffers.length);
            for (int c = 0; c < commandBuffers.length; c++) {
                cmds.setAtIndex(ADDRESS, c, commandBuffers[c]);
            }
            MemorySegment info = a.allocate(SUBMIT_INFO);
            si(info, SUBMIT_INFO, "sType", STYPE_SUBMIT_INFO);
            si(info, SUBMIT_INFO, "commandBufferCount", commandBuffers.length);
            sa(info, SUBMIT_INFO, "pCommandBuffers", cmds);
            if (waits.length + signals.length > 0) {
                MemorySegment timeline = a.allocate(TIMELINE_SEMAPHORE_SUBMIT_INFO);
                si(timeline, TIMELINE_SEMAPHORE_SUBMIT_INFO, "sType", STYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
                if (waits.length > 0) {
                    int[] stages = new int[waits.length];
                    java.util.Arrays.fill(stages, PIPELINE_STAGE_ALL_COMMANDS);
                    si(info, SUBMIT_INFO, "waitSemaphoreCount", waits.length);
                    sa(info, SUBMIT_INFO, "pWaitSemaphores", a.allocateFrom(JAVA_LONG, waits));
                    sa(info, SUBMIT_INFO, "pWaitDstStageMask", a.allocateFrom(JAVA_INT, stages));
                    si(timeline, TIMELINE_SEMAPHORE_SUBMIT_INFO, "waitSemaphoreValueCount", waits.length);
                    sa(timeline, TIMELINE_SEMAPHORE_SUBMIT_INFO, "pWaitSemaphoreValues",
                            a.allocateFrom(JAVA_LONG, waitValues));
                }
                if (signals.length > 0) {
                    si(info, SUBMIT_INFO, "signalSemaphoreCount", signals.length);
                    sa(info, SUBMIT_INFO, "pSignalSemaphores", a.allocateFrom(JAVA_LONG, signals));
                    si(timeline, TIMELINE_SEMAPHORE_SUBMIT_INFO, "signalSemaphoreValueCount", signals.length);
                    sa(timeline, TIMELINE_SEMAPHORE_SUBMIT_INFO, "pSignalSemaphoreValues",
                            a.allocateFrom(JAVA_LONG, signalValues));
                }
                sa(info, SUBMIT_INFO, "pNext", timeline);
            }
            check(invoke(vkQueueSubmit, queue, 1, info, fence), "vkQueueSubmit");
        }
    }

    // --- timestamps -----------------------------------------------------------------------------------

    long createTimestampPool(int queries) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(QUERY_POOL_CREATE_INFO);
            si(info, QUERY_POOL_CREATE_INFO, "sType", STYPE_QUERY_POOL_CREATE_INFO);
            si(info, QUERY_POOL_CREATE_INFO, "queryType", QUERY_TYPE_TIMESTAMP);
            si(info, QUERY_POOL_CREATE_INFO, "queryCount", queries);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateQueryPool, dev, info, MemorySegment.NULL, out), "vkCreateQueryPool");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroyQueryPool(long pool) {
        invokeVoid(vkDestroyQueryPool, dev, pool, MemorySegment.NULL);
    }

    void recordResetQueries(MemorySegment cmd, long pool, int first, int count) {
        invokeVoid(vkCmdResetQueryPool, cmd, pool, first, count);
    }

    /** A timestamp written to query {@code query} once every command before it has reached {@code stage}. */
    void recordTimestamp(MemorySegment cmd, int stage, long pool, int query) {
        invokeVoid(vkCmdWriteTimestamp, cmd, stage, pool, query);
    }

    /** Queries {@code first .. first + count - 1} as 64-bit ticks, or null when any is not written yet. */
    long[] timestamps(long pool, int first, int count) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(JAVA_LONG, count);
            int result = invoke(vkGetQueryPoolResults, dev, pool, first, count, (long) count * Long.BYTES, out,
                    (long) Long.BYTES, QUERY_RESULT_64);
            if (result == VK_NOT_READY) {
                return null;
            }
            check(result, "vkGetQueryPoolResults");
            return out.toArray(JAVA_LONG);
        }
    }

    // --- timeline semaphores ---------------------------------------------------------------------------

    /** A timeline semaphore starting at {@code initial}. The device must have been made with the feature. */
    long createTimeline(long initial) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment type = a.allocate(SEMAPHORE_TYPE_CREATE_INFO);
            si(type, SEMAPHORE_TYPE_CREATE_INFO, "sType", STYPE_SEMAPHORE_TYPE_CREATE_INFO);
            si(type, SEMAPHORE_TYPE_CREATE_INFO, "semaphoreType", SEMAPHORE_TYPE_TIMELINE);
            sl(type, SEMAPHORE_TYPE_CREATE_INFO, "initialValue", initial);
            MemorySegment info = a.allocate(SEMAPHORE_CREATE_INFO);
            si(info, SEMAPHORE_CREATE_INFO, "sType", STYPE_SEMAPHORE_CREATE_INFO);
            sa(info, SEMAPHORE_CREATE_INFO, "pNext", type);
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkCreateSemaphore, dev, info, MemorySegment.NULL, out), "vkCreateSemaphore");
            return out.get(JAVA_LONG, 0);
        }
    }

    void destroySemaphore(long semaphore) {
        invokeVoid(vkDestroySemaphore, dev, semaphore, MemorySegment.NULL);
    }

    /** The timeline's value now. */
    long timelineValue(long semaphore) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(JAVA_LONG);
            check(invoke(vkGetSemaphoreCounterValue, dev, semaphore, out), "vkGetSemaphoreCounterValue");
            return out.get(JAVA_LONG, 0);
        }
    }

    /** Blocks until the timeline reaches {@code value}, or {@code timeoutNanos} pass; whether it reached it. */
    boolean waitTimeline(long semaphore, long value, long timeoutNanos) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(SEMAPHORE_WAIT_INFO);
            si(info, SEMAPHORE_WAIT_INFO, "sType", STYPE_SEMAPHORE_WAIT_INFO);
            si(info, SEMAPHORE_WAIT_INFO, "semaphoreCount", 1);
            sa(info, SEMAPHORE_WAIT_INFO, "pSemaphores", a.allocateFrom(JAVA_LONG, semaphore));
            sa(info, SEMAPHORE_WAIT_INFO, "pValues", a.allocateFrom(JAVA_LONG, value));
            int result = invoke(vkWaitSemaphores, dev, info, timeoutNanos);
            if (result == VK_TIMEOUT) {
                return false;
            }
            check(result, "vkWaitSemaphores");
            return true;
        }
    }

    /** Sets the timeline to {@code value} from the host. It may only move forward. */
    void signalTimeline(long semaphore, long value) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment info = a.allocate(SEMAPHORE_SIGNAL_INFO);
            si(info, SEMAPHORE_SIGNAL_INFO, "sType", STYPE_SEMAPHORE_SIGNAL_INFO);
            sl(info, SEMAPHORE_SIGNAL_INFO, "semaphore", semaphore);
            sl(info, SEMAPHORE_SIGNAL_INFO, "value", value);
            check(invoke(vkSignalSemaphore, dev, info), "vkSignalSemaphore");
        }
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
