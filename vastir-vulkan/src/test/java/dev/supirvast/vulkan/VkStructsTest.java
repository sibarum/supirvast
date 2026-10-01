package dev.supirvast.vulkan;

import dev.supirvast.vulkan.VkStructs;
import org.junit.jupiter.api.Test;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout.PathElement;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pins the size of every layout in {@link VkStructs}, and the offset of the fields most easily got wrong.
 *
 * <h2>Why a test for constants</h2>
 *
 * <p>These layouts describe C structs that live on the other side of an FFI boundary, and nothing on this
 * side checks them. A missing {@code paddingLayout(4)} does not fail to compile, does not fail to allocate,
 * and does not fail at the call: the driver reads the bytes at the offsets its own header says, so a
 * mis-padded layout writes {@code width} where the driver looks for {@code height} — or writes a pointer
 * field half into a length field, which is the version that corrupts memory rather than merely drawing the
 * wrong picture.
 *
 * <p>That was survivable while each of these lived privately next to its one caller, because a mistake broke
 * one class. They are shared now, so a mistake breaks the windowed presenter, both offscreen paths and the
 * new {@code dev.vexelray.vulkan.present.OffscreenPresenter} at once — and the three GPU checks that would
 * catch it need a device, so on a machine without one nothing would.
 *
 * <p>The expected numbers are the sizes the Vulkan headers give on a 64-bit ABI, written as literals on
 * purpose. Deriving them from the layout would assert only that the layout equals itself.
 */
class VkStructsTest {

    @Test
    void everyLayoutIsTheSizeItsCStructIs() {
        assertEquals(24, VkStructs.CREATE_INFO.byteSize(), "VkFenceCreateInfo / VkSemaphoreCreateInfo");
        assertEquals(24, VkStructs.COMMAND_POOL_CREATE_INFO.byteSize(), "VkCommandPoolCreateInfo");
        assertEquals(32, VkStructs.COMMAND_BUFFER_ALLOCATE_INFO.byteSize(), "VkCommandBufferAllocateInfo");
        assertEquals(32, VkStructs.COMMAND_BUFFER_BEGIN_INFO.byteSize(), "VkCommandBufferBeginInfo");
        assertEquals(64, VkStructs.RENDER_PASS_BEGIN_INFO.byteSize(), "VkRenderPassBeginInfo");
        assertEquals(72, VkStructs.SUBMIT_INFO.byteSize(), "VkSubmitInfo");
        assertEquals(88, VkStructs.IMAGE_CREATE_INFO.byteSize(), "VkImageCreateInfo");
        assertEquals(80, VkStructs.IMAGE_VIEW_CREATE_INFO.byteSize(), "VkImageViewCreateInfo");
        assertEquals(64, VkStructs.FRAMEBUFFER_CREATE_INFO.byteSize(), "VkFramebufferCreateInfo");
        assertEquals(56, VkStructs.BUFFER_CREATE_INFO.byteSize(), "VkBufferCreateInfo");
        assertEquals(24, VkStructs.MEMORY_REQUIREMENTS.byteSize(), "VkMemoryRequirements");
        assertEquals(32, VkStructs.MEMORY_ALLOCATE_INFO.byteSize(), "VkMemoryAllocateInfo");
        assertEquals(56, VkStructs.BUFFER_IMAGE_COPY.byteSize(), "VkBufferImageCopy");
        assertEquals(24, VkStructs.VIEWPORT.byteSize(), "VkViewport");
        assertEquals(16, VkStructs.RECT_2D.byteSize(), "VkRect2D");
    }

    /**
     * The offsets a padding mistake actually moves.
     *
     * <p>Not every field: a size check already catches anything that changes the total, and a per-field
     * assertion for all of them would be the layout written twice. These are the ones where a struct could
     * stay the right size and still be wrong — the field right after a pointer, and the field right after an
     * odd number of 32-bit members, which is where an omitted or spurious pad hides.
     */
    @Test
    void theFieldsAfterEachPadIsWhereItShouldBe() {
        // pNext is eight-aligned, so sType is followed by four bytes of padding in every one of these.
        assertEquals(8, offset(VkStructs.CREATE_INFO, "pNext"));
        assertEquals(16, offset(VkStructs.COMMAND_POOL_CREATE_INFO, "flags"));

        // renderArea is a VkRect2D of two VkOffset2D/VkExtent2D ints each, flattened — four consecutive ints
        // after two 64-bit handles, and then clearValueCount, which is where the trailing pad belongs.
        assertEquals(24, offset(VkStructs.RENDER_PASS_BEGIN_INFO, "framebuffer"));
        assertEquals(32, offset(VkStructs.RENDER_PASS_BEGIN_INFO, "area_offset_x"));
        assertEquals(40, offset(VkStructs.RENDER_PASS_BEGIN_INFO, "area_extent_width"));
        assertEquals(56, offset(VkStructs.RENDER_PASS_BEGIN_INFO, "pClearValues"));

        // Three 32-bit extent members between two eight-aligned regions — the classic place to lose a pad.
        // 28, not 24: flags, imageType and format are three ints after the eight-byte pNext, so the extent
        // starts on a four-aligned odd boundary and no padding precedes it.
        assertEquals(28, offset(VkStructs.IMAGE_CREATE_INFO, "extent_width"));
        assertEquals(64, offset(VkStructs.IMAGE_CREATE_INFO, "queueFamilyIndexCount"));
        assertEquals(72, offset(VkStructs.IMAGE_CREATE_INFO, "pQueueFamilyIndices"));
        assertEquals(80, offset(VkStructs.IMAGE_CREATE_INFO, "initialLayout"));

        assertEquals(24, offset(VkStructs.IMAGE_VIEW_CREATE_INFO, "image"));
        assertEquals(40, offset(VkStructs.IMAGE_VIEW_CREATE_INFO, "c_r"));
        assertEquals(56, offset(VkStructs.IMAGE_VIEW_CREATE_INFO, "sr_aspectMask"));

        assertEquals(24, offset(VkStructs.FRAMEBUFFER_CREATE_INFO, "renderPass"));
        assertEquals(40, offset(VkStructs.FRAMEBUFFER_CREATE_INFO, "pAttachments"));
        assertEquals(48, offset(VkStructs.FRAMEBUFFER_CREATE_INFO, "width"));

        assertEquals(24, offset(VkStructs.BUFFER_CREATE_INFO, "size"));
        assertEquals(48, offset(VkStructs.BUFFER_CREATE_INFO, "pQueueFamilyIndices"));

        assertEquals(16, offset(VkStructs.MEMORY_ALLOCATE_INFO, "allocationSize"));
        assertEquals(16, offset(VkStructs.MEMORY_REQUIREMENTS, "memoryTypeBits"));

        assertEquals(16, offset(VkStructs.BUFFER_IMAGE_COPY, "is_aspectMask"));
        assertEquals(44, offset(VkStructs.BUFFER_IMAGE_COPY, "ext_width"));

        assertEquals(48, offset(VkStructs.SUBMIT_INFO, "pCommandBuffers"));
        assertEquals(64, offset(VkStructs.SUBMIT_INFO, "pSignalSemaphores"));

        assertEquals(8, offset(VkStructs.VIEWPORT, "width"));
        assertEquals(20, offset(VkStructs.VIEWPORT, "maxDepth"));
        assertEquals(8, offset(VkStructs.RECT_2D, "extent_width"));
    }

    private static long offset(GroupLayout layout, String field) {
        return layout.byteOffset(PathElement.groupElement(field));
    }
}
