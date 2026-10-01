package dev.supirvast.vulkan;

/**
 * Vulkan enum/flag/structure-type constants used by the bindings. Grouped here so magic numbers never appear in
 * struct-filling code (the values are from the Vulkan spec; kept minimal — only what VexelRay uses so far).
 */
public final class Vk {

    private Vk() {
    }

    public static final int VK_SUCCESS = 0;
    public static final int VK_TRUE = 1;

    // Structure types
    public static final int STRUCTURE_TYPE_SUBMIT_INFO = 4;
    public static final int STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
    public static final int STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
    public static final int STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
    public static final int STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14;
    public static final int STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
    public static final int STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
    public static final int STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
    public static final int STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45;

    // Formats
    public static final int FORMAT_R8G8B8A8_UNORM = 37;

    // Image
    public static final int IMAGE_TYPE_2D = 1;
    public static final int IMAGE_TILING_OPTIMAL = 0;
    public static final int SAMPLE_COUNT_1_BIT = 1;
    public static final int IMAGE_USAGE_TRANSFER_SRC_BIT = 0x0001;
    public static final int IMAGE_USAGE_TRANSFER_DST_BIT = 0x0002;
    public static final int IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x0010;
    public static final int IMAGE_ASPECT_COLOR_BIT = 0x0001;

    // Layouts
    public static final int IMAGE_LAYOUT_UNDEFINED = 0;
    public static final int IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6;
    public static final int IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7;

    // Buffer
    public static final int BUFFER_USAGE_TRANSFER_DST_BIT = 0x0002;

    // Sharing
    public static final int SHARING_MODE_EXCLUSIVE = 0;

    // Memory properties
    public static final int MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x0001;
    public static final int MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x0002;
    public static final int MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x0004;

    // Command buffer
    public static final int COMMAND_BUFFER_LEVEL_PRIMARY = 0;
    public static final int COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x0001;

    // Pipeline stages / access
    public static final int PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x0001;
    public static final int PIPELINE_STAGE_TRANSFER_BIT = 0x1000;
    public static final int ACCESS_TRANSFER_READ_BIT = 0x0800;
    public static final int ACCESS_TRANSFER_WRITE_BIT = 0x1000;

    // Sentinels
    public static final int QUEUE_FAMILY_IGNORED = ~0;
    public static final int SUBPASS_EXTERNAL = ~0;
    public static final long WHOLE_SIZE = ~0L;

    // --- graphics-pipeline structure types ---
    public static final int STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
    public static final int STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
    public static final int STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
    public static final int STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 19;
    public static final int STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 20;
    public static final int STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 22;
    public static final int STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 23;
    public static final int STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 24;
    public static final int STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 26;
    public static final int STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 27;
    public static final int STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28;
    /** VkDynamicState: viewport and scissor are set per-frame via vkCmdSetViewport/Scissor (resize support). */
    public static final int DYNAMIC_STATE_VIEWPORT = 0;
    public static final int DYNAMIC_STATE_SCISSOR = 1;
    public static final int STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
    public static final int STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 37;
    public static final int STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 38;
    public static final int STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 43;

    // Image view / attachments
    public static final int IMAGE_VIEW_TYPE_2D = 1;
    public static final int IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 2;
    public static final int ATTACHMENT_LOAD_OP_CLEAR = 1;
    public static final int ATTACHMENT_STORE_OP_STORE = 0;
    public static final int ATTACHMENT_LOAD_OP_DONT_CARE = 2;
    public static final int ATTACHMENT_STORE_OP_DONT_CARE = 1;

    // Pipeline
    public static final int PIPELINE_BIND_POINT_GRAPHICS = 0;
    public static final int SHADER_STAGE_VERTEX_BIT = 0x0001;
    public static final int SHADER_STAGE_FRAGMENT_BIT = 0x0010;
    public static final int PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3;
    public static final int POLYGON_MODE_FILL = 0;
    public static final int CULL_MODE_NONE = 0;
    public static final int FRONT_FACE_COUNTER_CLOCKWISE = 0;
    public static final int COLOR_COMPONENT_RGBA =
            0x1 | 0x2 | 0x4 | 0x8;
    public static final int SUBPASS_CONTENTS_INLINE = 0;

    // Stages / access for the color-attachment dependency
    public static final int PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x0400;
    public static final int PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x2000;
    public static final int ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x0100;

    // --- swapchain / presentation (VK_KHR_swapchain, VK_KHR_surface) ---
    public static final int STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9;
    public static final int STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000;
    public static final int STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001;

    public static final int FORMAT_B8G8R8A8_UNORM = 44;
    public static final int FORMAT_B8G8R8A8_SRGB = 50;
    public static final int COLOR_SPACE_SRGB_NONLINEAR_KHR = 0;
    public static final int PRESENT_MODE_FIFO_KHR = 2;
    public static final int COMPOSITE_ALPHA_OPAQUE_BIT_KHR = 0x0001;
    public static final int IMAGE_LAYOUT_PRESENT_SRC_KHR = 1000001002;

    public static final int SUBOPTIMAL_KHR = 1000001003;
    public static final int ERROR_OUT_OF_DATE_KHR = -1000001004;
    /**
     * VkResult: the device is gone — a GPU reset, a driver crash, a hot-unplugged eGPU. Unrecoverable at the
     * call site and recoverable only by rebuilding everything made from the device, which is why
     * {@link Ffm#check} raises it as its own exception type rather than as one more failed call.
     */
    public static final int ERROR_DEVICE_LOST = -4;

    // Command pool flags
    public static final int COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x0002;
    /** {@code 0xFFFFFFFF} — a surface {@code currentExtent} of this width means "the app chooses the size". */
    public static final int EXTENT_UNDEFINED = 0xFFFFFFFF;

    // --- sampled textures, descriptor sets, vertex buffers, alpha blending (MSDF text path) ---

    // Structure types
    public static final int STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 31;
    public static final int STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
    public static final int STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
    public static final int STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
    public static final int STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;

    // Formats (vertex attributes)
    public static final int FORMAT_R32_SFLOAT = 100;
    public static final int FORMAT_R32G32_SFLOAT = 103;
    public static final int FORMAT_R32G32B32A32_SFLOAT = 109;

    // Image usage / layout for a sampled image
    public static final int IMAGE_USAGE_SAMPLED_BIT = 0x0004;
    public static final int IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL = 5;

    // Buffer usage
    public static final int BUFFER_USAGE_TRANSFER_SRC_BIT = 0x0001;
    public static final int BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x0080;
    public static final int BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x0020;

    // Sampler
    public static final int FILTER_LINEAR = 1;
    public static final int SAMPLER_MIPMAP_MODE_NEAREST = 0;
    public static final int SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2;
    public static final int BORDER_COLOR_FLOAT_OPAQUE_BLACK = 0;

    // Descriptors
    public static final int DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1;
    public static final int DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;

    // Vertex input
    public static final int VERTEX_INPUT_RATE_VERTEX = 0;

    // Alpha blending
    public static final int BLEND_FACTOR_ZERO = 0;
    public static final int BLEND_FACTOR_ONE = 1;
    public static final int BLEND_FACTOR_SRC_ALPHA = 6;
    public static final int BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7;
    public static final int BLEND_OP_ADD = 0;

    // Pipeline stages / access for the sampled-image transition
    public static final int PIPELINE_STAGE_FRAGMENT_SHADER_BIT = 0x0080;
    public static final int ACCESS_SHADER_READ_BIT = 0x0020;

    // Depth. The shared depth attachment is what lets two techniques occlude each other instead of merely
    // painting over one another, so these arrive together with DepthAttachment and the render pass's second slot.
    public static final int FORMAT_D32_SFLOAT = 126;
    public static final int FORMAT_D24_UNORM_S8_UINT = 129;
    public static final int FORMAT_D32_SFLOAT_S8_UINT = 130;
    public static final int IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3;
    public static final int IMAGE_ASPECT_DEPTH_BIT = 0x0002;
    public static final int IMAGE_ASPECT_STENCIL_BIT = 0x0004;
    public static final int IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x0020;
    public static final int ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x0200;
    public static final int ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x0400;
    // Depth is written in the fragment-test stages, not COLOR_ATTACHMENT_OUTPUT. A subpass dependency that
    // names only the colour stage lets the next frame's depth clear race this frame's depth writes, which reads
    // as intermittent z-fighting on geometry that is not moving — so both tests belong in the mask.
    public static final int PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x0100;
    public static final int PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT = 0x0200;
    public static final int STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 25;
    public static final int COMPARE_OP_NEVER = 0;
    public static final int COMPARE_OP_LESS = 1;
    public static final int COMPARE_OP_LESS_OR_EQUAL = 3;
    public static final int COMPARE_OP_ALWAYS = 7;
}
