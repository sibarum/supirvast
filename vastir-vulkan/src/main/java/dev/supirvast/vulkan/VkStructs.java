package dev.supirvast.vulkan;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The {@code VkStruct} memory layouts more than one class in this runtime needs — declared once, with one
 * spelling per field.
 *
 * <p>These were private constants in {@code WindowedPresenter}, {@code OffscreenDraw} and
 * {@code OffscreenRenderer}, three independent transcriptions of the same C structs. Three copies of a padded
 * layout is not merely repetition: the field names had already drifted apart ({@code area_w} in one,
 * {@code area_extent_width} in another, for the same {@code VkRenderPassBeginInfo.renderArea.extent.width}),
 * so a reader comparing two of them could not tell whether the difference was a naming choice or a bug. And a
 * padding mistake in one copy is invisible in the other two: the loader reads whatever is at the offset, so
 * the symptom is a wrong number deep inside a driver rather than an error at the seam.
 *
 * <h2>Naming</h2>
 *
 * <p>Nested structs are flattened with an underscore path that names the real nesting, so a field can be
 * checked against the Vulkan header without decoding an abbreviation: {@code VkRenderPassBeginInfo.renderArea}
 * (a {@code VkRect2D} of a {@code VkOffset2D} and a {@code VkExtent2D}) is {@code area_offset_x},
 * {@code area_offset_y}, {@code area_extent_width}, {@code area_extent_height}. Flattened rather than nested
 * layouts because {@link Ffm}'s accessors take one field name, and a one-level path is what keeps the call
 * sites readable.
 *
 * <p>Every layout here is 64-bit: {@code ADDRESS} is eight bytes and the explicit
 * {@link MemoryLayout#paddingLayout} calls are the alignment a C compiler would insert. Non-dispatchable
 * handles ({@code VkImage}, {@code VkBuffer}, {@code VkRenderPass}) are {@code uint64} and so
 * {@code JAVA_LONG}; dispatchable ones ({@code VkDevice}, {@code VkCommandBuffer}) are pointers and so
 * {@code ADDRESS} — see {@code docs/native-bindings.md} §4.3.
 */
public final class VkStructs {

    private VkStructs() {
    }

    /**
     * The shape shared by {@code VkSemaphoreCreateInfo} and {@code VkFenceCreateInfo} — sType, pNext, flags.
     *
     * <p>One layout for two structs because they are the same three fields in the same order, and Vulkan says
     * so: this is not a coincidence being exploited, it is {@code VkFenceCreateInfo} and
     * {@code VkSemaphoreCreateInfo} being defined identically apart from their {@code sType} value.
     */
    public static final GroupLayout CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4)).withName("VkFenceCreateInfo");

    public static final GroupLayout COMMAND_POOL_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("queueFamilyIndex")
    ).withName("VkCommandPoolCreateInfo");

    public static final GroupLayout COMMAND_BUFFER_ALLOCATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("commandPool"), JAVA_INT.withName("level"), JAVA_INT.withName("commandBufferCount")
    ).withName("VkCommandBufferAllocateInfo");

    public static final GroupLayout COMMAND_BUFFER_BEGIN_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pInheritanceInfo")
    ).withName("VkCommandBufferBeginInfo");

    public static final GroupLayout RENDER_PASS_BEGIN_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("renderPass"), JAVA_LONG.withName("framebuffer"),
            JAVA_INT.withName("area_offset_x"), JAVA_INT.withName("area_offset_y"),
            JAVA_INT.withName("area_extent_width"), JAVA_INT.withName("area_extent_height"),
            JAVA_INT.withName("clearValueCount"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pClearValues")
    ).withName("VkRenderPassBeginInfo");

    public static final GroupLayout SUBMIT_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("waitSemaphoreCount"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pWaitSemaphores"),
            ADDRESS.withName("pWaitDstStageMask"), JAVA_INT.withName("commandBufferCount"),
            MemoryLayout.paddingLayout(4), ADDRESS.withName("pCommandBuffers"),
            JAVA_INT.withName("signalSemaphoreCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pSignalSemaphores")).withName("VkSubmitInfo");

    public static final GroupLayout IMAGE_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), JAVA_INT.withName("imageType"), JAVA_INT.withName("format"),
            JAVA_INT.withName("extent_width"), JAVA_INT.withName("extent_height"), JAVA_INT.withName("extent_depth"),
            JAVA_INT.withName("mipLevels"), JAVA_INT.withName("arrayLayers"), JAVA_INT.withName("samples"),
            JAVA_INT.withName("tiling"), JAVA_INT.withName("usage"), JAVA_INT.withName("sharingMode"),
            JAVA_INT.withName("queueFamilyIndexCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pQueueFamilyIndices"), JAVA_INT.withName("initialLayout"), MemoryLayout.paddingLayout(4)
    ).withName("VkImageCreateInfo");

    public static final GroupLayout IMAGE_VIEW_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("image"),
            JAVA_INT.withName("viewType"), JAVA_INT.withName("format"),
            JAVA_INT.withName("c_r"), JAVA_INT.withName("c_g"), JAVA_INT.withName("c_b"), JAVA_INT.withName("c_a"),
            JAVA_INT.withName("sr_aspectMask"), JAVA_INT.withName("sr_baseMipLevel"),
            JAVA_INT.withName("sr_levelCount"), JAVA_INT.withName("sr_baseArrayLayer"),
            JAVA_INT.withName("sr_layerCount"), MemoryLayout.paddingLayout(4)
    ).withName("VkImageViewCreateInfo");

    public static final GroupLayout FRAMEBUFFER_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("renderPass"),
            JAVA_INT.withName("attachmentCount"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pAttachments"),
            JAVA_INT.withName("width"), JAVA_INT.withName("height"), JAVA_INT.withName("layers"),
            MemoryLayout.paddingLayout(4)).withName("VkFramebufferCreateInfo");

    public static final GroupLayout BUFFER_CREATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_INT.withName("flags"), MemoryLayout.paddingLayout(4), JAVA_LONG.withName("size"),
            JAVA_INT.withName("usage"), JAVA_INT.withName("sharingMode"),
            JAVA_INT.withName("queueFamilyIndexCount"), MemoryLayout.paddingLayout(4),
            ADDRESS.withName("pQueueFamilyIndices")).withName("VkBufferCreateInfo");

    public static final GroupLayout MEMORY_REQUIREMENTS = MemoryLayout.structLayout(
            JAVA_LONG.withName("size"), JAVA_LONG.withName("alignment"),
            JAVA_INT.withName("memoryTypeBits"), MemoryLayout.paddingLayout(4)).withName("VkMemoryRequirements");

    public static final GroupLayout MEMORY_ALLOCATE_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("sType"), MemoryLayout.paddingLayout(4), ADDRESS.withName("pNext"),
            JAVA_LONG.withName("allocationSize"), JAVA_INT.withName("memoryTypeIndex"), MemoryLayout.paddingLayout(4)
    ).withName("VkMemoryAllocateInfo");

    public static final GroupLayout BUFFER_IMAGE_COPY = MemoryLayout.structLayout(
            JAVA_LONG.withName("bufferOffset"), JAVA_INT.withName("bufferRowLength"),
            JAVA_INT.withName("bufferImageHeight"),
            JAVA_INT.withName("is_aspectMask"), JAVA_INT.withName("is_mipLevel"),
            JAVA_INT.withName("is_baseArrayLayer"), JAVA_INT.withName("is_layerCount"),
            JAVA_INT.withName("off_x"), JAVA_INT.withName("off_y"), JAVA_INT.withName("off_z"),
            JAVA_INT.withName("ext_width"), JAVA_INT.withName("ext_height"), JAVA_INT.withName("ext_depth")
    ).withName("VkBufferImageCopy");

    public static final GroupLayout VIEWPORT = MemoryLayout.structLayout(
            JAVA_FLOAT.withName("x"), JAVA_FLOAT.withName("y"), JAVA_FLOAT.withName("width"),
            JAVA_FLOAT.withName("height"), JAVA_FLOAT.withName("minDepth"), JAVA_FLOAT.withName("maxDepth")
    ).withName("VkViewport");

    public static final GroupLayout RECT_2D = MemoryLayout.structLayout(
            JAVA_INT.withName("offset_x"), JAVA_INT.withName("offset_y"),
            JAVA_INT.withName("extent_width"), JAVA_INT.withName("extent_height")).withName("VkRect2D");
}
