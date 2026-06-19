package dev.supirvast.vastir.preview;

import dev.supirvast.vastir.tools.GraphicsPipelineSpec;
import dev.supirvast.vastir.tools.GraphicsPipelineSpec.VertexAttribute;
import dev.supirvast.vastir.type.Type;
import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.lwjgl.glfw.GLFW.GLFW_CLIENT_API;
import static org.lwjgl.glfw.GLFW.GLFW_FALSE;
import static org.lwjgl.glfw.GLFW.GLFW_NO_API;
import static org.lwjgl.glfw.GLFW.GLFW_VISIBLE;
import static org.lwjgl.glfw.GLFW.glfwCreateWindow;
import static org.lwjgl.glfw.GLFW.glfwDestroyWindow;
import static org.lwjgl.glfw.GLFW.glfwGetTime;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFW.glfwWindowHint;
import static org.lwjgl.glfw.GLFW.glfwWindowShouldClose;
import static org.lwjgl.glfw.GLFWVulkan.glfwCreateWindowSurface;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.glfw.GLFWVulkan.glfwVulkanSupported;
import static org.lwjgl.stb.STBImage.stbi_image_free;
import static org.lwjgl.stb.STBImage.stbi_load;
import static org.lwjgl.stb.STBImageWrite.stbi_write_png;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The windowed Vulkan previewer. As of step 4 it draws a model: it loads the vertex + fragment SPIR-V and an
 * OBJ mesh from the CLI, builds a graphics pipeline from a {@link GraphicsPipelineSpec} (vertex input state
 * from the spec's layout), uploads interleaved {@code position+normal} vertices to a buffer, and issues an
 * indexed draw into a depth-tested color attachment each frame.
 *
 * <p>v1 renders <em>statically</em>: the vertex shader emits {@code gl_Position} straight from the model
 * position (the sample model is authored pre-rotated in valid clip space), with the depth buffer doing real
 * hidden-surface removal. A runtime MVP transform / auto-rotation is deferred — it needs a {@code Matrix} type
 * and push-constant support in the {@code core} IR, which don't exist yet.
 *
 * <p>Resource ownership and teardown mirror {@code GpuContext} in vastir-tools: a {@code check()} guards every
 * {@code VkResult}, {@link MemoryStack} scopes native allocations, and {@link #close()} destroys in reverse
 * creation order. Windows-only (matching the project's native story); requires a Vulkan 1.3 device.
 */
public final class PreviewApp implements AutoCloseable {

    // A recognizable, non-black clear color (cornflower-ish) so the rendered model stands out.
    private static final float[] CLEAR_RGBA = {0.39f, 0.58f, 0.93f, 1.0f};
    private static final int DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
    // Push-constant block: mat4 mvp (offset 0), mat4 model (64), vec3 cameraPosition (128) — 144 bytes.
    private static final int PUSH_BYTES = 64 + 64 + 16;
    private static final float[] CAMERA_EYE = {0.0f, 2.2f, 5.5f};

    private final PreviewOptions options;

    private long window;
    private VkInstance instance;
    private long surface;
    private VkPhysicalDevice physical;
    private VkDevice device;
    private int queueFamily;
    private VkQueue queue;

    private long swapchain;
    private int swapchainFormat;
    private int extentWidth;
    private int extentHeight;
    private long[] swapchainImages = new long[0];
    private long[] imageViews = new long[0];
    private long[] framebuffers = new long[0];
    private long renderPass;

    private long depthImage;
    private long depthMemory;
    private long depthView;

    private long vertexShaderModule;
    private long fragmentShaderModule;
    private long pipelineLayout;
    private long graphicsPipeline;

    // Combined image-sampler textures (one descriptor set holding all of them), parallel arrays by index.
    private int[] textureBindings = new int[0];
    private long[] textureImages = new long[0];
    private long[] textureMemories = new long[0];
    private long[] textureViews = new long[0];
    private long[] textureSamplers = new long[0];
    private long descriptorSetLayout;
    private long descriptorPool;
    private long descriptorSet;

    private long vertexBuffer;
    private long vertexMemory;
    private long indexBuffer;
    private long indexMemory;
    private int indexCount;

    private long commandPool;
    private VkCommandBuffer commandBuffer;

    private long imageAvailable;
    private long renderFinished;
    private long inFlight;

    public PreviewApp(PreviewOptions options) {
        this.options = options;
    }

    public static void main(String[] args) {
        PreviewOptions options;
        try {
            options = PreviewOptions.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.println(PreviewOptions.usage());
            System.exit(2);
            return;
        }
        try (PreviewApp app = new PreviewApp(options)) {
            app.run();
        }
    }

    /** Initializes window + Vulkan, then either captures a one-shot screenshot or runs the render loop. */
    public void run() {
        initWindow();
        initVulkan();
        if (options.screenshot().isPresent()) {
            captureScreenshot(options.screenshot().get());
        } else {
            loop();
        }
    }

    private void initWindow() {
        if (!glfwInit()) {
            throw new IllegalStateException("failed to initialize GLFW");
        }
        if (!glfwVulkanSupported()) {
            throw new IllegalStateException("GLFW reports Vulkan is not supported on this machine");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // no OpenGL context; this is a Vulkan window
        if (options.screenshot().isPresent()) {
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // one-shot capture: render off-screen, don't flash a window
        }
        window = glfwCreateWindow(options.width(), options.height(), "vastir-preview", NULL, NULL);
        if (window == NULL) {
            throw new IllegalStateException("failed to create the GLFW window");
        }
    }

    private void initVulkan() {
        GraphicsPipelineSpec spec = GraphicsPipelineSpec.standard(
                readBytes(options.vert()), readBytes(options.frag()));
        Mesh mesh = ModelLoader.load(options.model());
        try (MemoryStack stack = stackPush()) {
            createInstance(stack);
            createSurface(stack);
            pickPhysicalDevice(stack);
            createDevice(stack);
            createSwapchain(stack);
            createDepthResources(stack);
            createRenderPass(stack);
            createFramebuffers(stack);
            createCommandPoolAndBuffer(stack);   // before textures (upload needs the pool) and the pipeline
            createTextureResources(stack);       // descriptor set layout must exist before the pipeline layout
            createGraphicsPipeline(spec, stack);
            createVertexAndIndexBuffers(mesh, stack);
            createSyncObjects(stack);
        }
    }

    private void createInstance(MemoryStack stack) {
        PointerBuffer glfwExtensions = glfwGetRequiredInstanceExtensions();
        if (glfwExtensions == null) {
            throw new IllegalStateException("GLFW could not enumerate the required Vulkan instance extensions");
        }
        VkApplicationInfo app = VkApplicationInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                .pApplicationName(stack.UTF8("vastir-preview"))
                .apiVersion(VK_API_VERSION_1_3);
        VkInstanceCreateInfo info = VkInstanceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                .pApplicationInfo(app)
                .ppEnabledExtensionNames(glfwExtensions);
        PointerBuffer pInstance = stack.mallocPointer(1);
        check(vkCreateInstance(info, null, pInstance), "vkCreateInstance");
        instance = new VkInstance(pInstance.get(0), info);
    }

    private void createSurface(MemoryStack stack) {
        LongBuffer pSurface = stack.mallocLong(1);
        check(glfwCreateWindowSurface(instance, window, null, pSurface), "glfwCreateWindowSurface");
        surface = pSurface.get(0);
    }

    private void pickPhysicalDevice(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices");
        if (count.get(0) == 0) {
            throw new IllegalStateException("no Vulkan physical devices found");
        }
        PointerBuffer devices = stack.mallocPointer(count.get(0));
        check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices");
        for (int i = 0; i < devices.capacity(); i++) {
            VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
            int family = findGraphicsPresentQueueFamily(candidate, stack);
            if (family >= 0) {
                physical = candidate;
                queueFamily = family;
                return;
            }
        }
        throw new IllegalStateException("no device with a graphics + present queue family for this surface");
    }

    /** A queue family that supports both graphics and presentation to our surface (the common case). */
    private int findGraphicsPresentQueueFamily(VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, families);
        IntBuffer presentSupport = stack.mallocInt(1);
        for (int i = 0; i < families.capacity(); i++) {
            boolean graphics = (families.get(i).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, presentSupport),
                    "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (graphics && presentSupport.get(0) == VK_TRUE) {
                return i;
            }
        }
        return -1;
    }

    private void createDevice(MemoryStack stack) {
        VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                .queueFamilyIndex(queueFamily)
                .pQueuePriorities(stack.floats(1.0f));
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pQueueCreateInfos(queues)
                .ppEnabledExtensionNames(stack.pointers(stack.UTF8(VK_KHR_SWAPCHAIN_EXTENSION_NAME)));
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        device = new VkDevice(pDevice.get(0), physical, info);

        PointerBuffer pQueue = stack.mallocPointer(1);
        vkGetDeviceQueue(device, queueFamily, 0, pQueue);
        queue = new VkQueue(pQueue.get(0), device);
    }

    private void createSwapchain(MemoryStack stack) {
        VkSurfaceCapabilitiesKHR caps = VkSurfaceCapabilitiesKHR.malloc(stack);
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        VkSurfaceFormatKHR.Buffer formats = surfaceFormats(stack);
        VkSurfaceFormatKHR chosen = chooseSurfaceFormat(formats);
        swapchainFormat = chosen.format();

        chooseExtent(caps);
        int imageCount = caps.minImageCount() + 1;
        if (caps.maxImageCount() > 0 && imageCount > caps.maxImageCount()) {
            imageCount = caps.maxImageCount();
        }

        VkSwapchainCreateInfoKHR info = VkSwapchainCreateInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
                .surface(surface)
                .minImageCount(imageCount)
                .imageFormat(swapchainFormat)
                .imageColorSpace(chosen.colorSpace())
                .imageArrayLayers(1)
                // TRANSFER_SRC so a later step can copy a presented image out for a PNG screenshot.
                .imageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .imageSharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .preTransform(caps.currentTransform())
                .compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                .presentMode(VK_PRESENT_MODE_FIFO_KHR)
                .clipped(true)
                .oldSwapchain(VK_NULL_HANDLE);
        info.imageExtent().width(extentWidth).height(extentHeight);

        LongBuffer pSwapchain = stack.mallocLong(1);
        check(vkCreateSwapchainKHR(device, info, null, pSwapchain), "vkCreateSwapchainKHR");
        swapchain = pSwapchain.get(0);

        swapchainImages = swapchainImages(stack);
        createImageViews(stack, swapchainImages);
    }

    private VkSurfaceFormatKHR.Buffer surfaceFormats(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, null),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        VkSurfaceFormatKHR.Buffer formats = VkSurfaceFormatKHR.malloc(count.get(0), stack);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, formats),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        return formats;
    }

    /** Prefer 8-bit BGRA UNORM/SRGB; otherwise take whatever the surface offers first. */
    private static VkSurfaceFormatKHR chooseSurfaceFormat(VkSurfaceFormatKHR.Buffer formats) {
        for (int i = 0; i < formats.capacity(); i++) {
            VkSurfaceFormatKHR f = formats.get(i);
            boolean bgra = f.format() == VK_FORMAT_B8G8R8A8_UNORM || f.format() == VK_FORMAT_B8G8R8A8_SRGB;
            if (bgra && f.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return f;
            }
        }
        return formats.get(0);
    }

    private void chooseExtent(VkSurfaceCapabilitiesKHR caps) {
        VkExtent2D current = caps.currentExtent();
        if (current.width() != 0xFFFFFFFF) {   // the surface dictates the size
            extentWidth = current.width();
            extentHeight = current.height();
            return;
        }
        // Otherwise clamp the requested size to the surface's allowed range.
        extentWidth = clamp(options.width(), caps.minImageExtent().width(), caps.maxImageExtent().width());
        extentHeight = clamp(options.height(), caps.minImageExtent().height(), caps.maxImageExtent().height());
    }

    private long[] swapchainImages(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetSwapchainImagesKHR(device, swapchain, count, null), "vkGetSwapchainImagesKHR");
        LongBuffer pImages = stack.mallocLong(count.get(0));
        check(vkGetSwapchainImagesKHR(device, swapchain, count, pImages), "vkGetSwapchainImagesKHR");
        long[] images = new long[count.get(0)];
        pImages.get(images);
        return images;
    }

    private void createImageViews(MemoryStack stack, long[] images) {
        imageViews = new long[images.length];
        for (int i = 0; i < images.length; i++) {
            imageViews[i] = createImageView(images[i], swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT, stack);
        }
    }

    private long createImageView(long image, int format, int aspect, MemoryStack stack) {
        VkImageViewCreateInfo info = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(image)
                .viewType(VK_IMAGE_VIEW_TYPE_2D)
                .format(format);
        info.subresourceRange()
                .aspectMask(aspect)
                .baseMipLevel(0).levelCount(1)
                .baseArrayLayer(0).layerCount(1);
        LongBuffer pView = stack.mallocLong(1);
        check(vkCreateImageView(device, info, null, pView), "vkCreateImageView");
        return pView.get(0);
    }

    private void createDepthResources(MemoryStack stack) {
        VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D)
                .format(DEPTH_FORMAT)
                .mipLevels(1)
                .arrayLayers(1)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        info.extent().width(extentWidth).height(extentHeight).depth(1);
        LongBuffer pImage = stack.mallocLong(1);
        check(vkCreateImage(device, info, null, pImage), "vkCreateImage(depth)");
        depthImage = pImage.get(0);

        VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, depthImage, requirements);
        depthMemory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, stack);
        check(vkBindImageMemory(device, depthImage, depthMemory, 0), "vkBindImageMemory(depth)");

        depthView = createImageView(depthImage, DEPTH_FORMAT, VK_IMAGE_ASPECT_DEPTH_BIT, stack);
    }

    private void createRenderPass(MemoryStack stack) {
        VkAttachmentDescription.Buffer attachments = VkAttachmentDescription.calloc(2, stack);
        attachments.get(0)   // color
                .format(swapchainFormat)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE)
                .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .finalLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        attachments.get(1)   // depth
                .format(DEPTH_FORMAT)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .finalLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        VkAttachmentReference.Buffer colorRef = VkAttachmentReference.calloc(1, stack)
                .attachment(0)
                .layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkAttachmentReference depthRef = VkAttachmentReference.calloc(stack)
                .attachment(1)
                .layout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        VkSubpassDescription.Buffer subpass = VkSubpassDescription.calloc(1, stack)
                .pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS)
                .colorAttachmentCount(1)
                .pColorAttachments(colorRef)
                .pDepthStencilAttachment(depthRef);

        // Wait for the color + depth stages of prior work before this subpass writes them.
        VkSubpassDependency.Buffer dependency = VkSubpassDependency.calloc(1, stack);
        dependency.get(0)
                .srcSubpass(VK_SUBPASS_EXTERNAL)
                .dstSubpass(0)
                .srcStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
                .srcAccessMask(0)
                .dstStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
                .dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkRenderPassCreateInfo info = VkRenderPassCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO)
                .pAttachments(attachments)
                .pSubpasses(subpass)
                .pDependencies(dependency);
        LongBuffer pRenderPass = stack.mallocLong(1);
        check(vkCreateRenderPass(device, info, null, pRenderPass), "vkCreateRenderPass");
        renderPass = pRenderPass.get(0);
    }

    private void createFramebuffers(MemoryStack stack) {
        framebuffers = new long[imageViews.length];
        for (int i = 0; i < imageViews.length; i++) {
            VkFramebufferCreateInfo info = VkFramebufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO)
                    .renderPass(renderPass)
                    .pAttachments(stack.longs(imageViews[i], depthView))   // color + shared depth
                    .width(extentWidth)
                    .height(extentHeight)
                    .layers(1);
            LongBuffer pFramebuffer = stack.mallocLong(1);
            check(vkCreateFramebuffer(device, info, null, pFramebuffer), "vkCreateFramebuffer");
            framebuffers[i] = pFramebuffer.get(0);
        }
    }

    private void createGraphicsPipeline(GraphicsPipelineSpec spec, MemoryStack stack) {
        vertexShaderModule = createShaderModule(spec.vertexSpirv(), stack);
        fragmentShaderModule = createShaderModule(spec.fragmentSpirv(), stack);

        VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
        stages.get(0)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_VERTEX_BIT)
                .module(vertexShaderModule)
                .pName(stack.UTF8(spec.vertexEntryPoint()));
        stages.get(1)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_FRAGMENT_BIT)
                .module(fragmentShaderModule)
                .pName(stack.UTF8(spec.fragmentEntryPoint()));

        VkVertexInputBindingDescription.Buffer binding = VkVertexInputBindingDescription.calloc(1, stack)
                .binding(0)
                .stride(spec.vertexStrideBytes())
                .inputRate(VK_VERTEX_INPUT_RATE_VERTEX);
        List<VertexAttribute> layout = spec.vertexLayout();
        VkVertexInputAttributeDescription.Buffer attributes =
                VkVertexInputAttributeDescription.calloc(layout.size(), stack);
        for (int i = 0; i < layout.size(); i++) {
            VertexAttribute attribute = layout.get(i);
            attributes.get(i)
                    .location(attribute.location())
                    .binding(0)
                    .format(vertexFormat(attribute.type()))
                    .offset(attribute.offsetBytes());
        }
        VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO)
                .pVertexBindingDescriptions(binding)
                .pVertexAttributeDescriptions(attributes);

        VkPipelineInputAssemblyStateCreateInfo inputAssembly =
                VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                        .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                        .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        // Flip Y (origin at bottom, negative height) so model-space +Y is screen-up — Vulkan clip space is
        // Y-down by default, which would otherwise render everything (and its lighting) upside down. Core since
        // Vulkan 1.1; we run 1.3.
        VkViewport.Buffer viewport = VkViewport.calloc(1, stack)
                .x(0).y(extentHeight).width(extentWidth).height(-extentHeight).minDepth(0).maxDepth(1);
        VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack);
        scissor.get(0).offset().set(0, 0);
        scissor.get(0).extent().width(extentWidth).height(extentHeight);
        VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                .pViewports(viewport)
                .pScissors(scissor);

        VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                .polygonMode(VK_POLYGON_MODE_FILL)
                // No culling: an arbitrary OBJ's winding is unknown, and depth still resolves overlap correctly.
                .cullMode(VK_CULL_MODE_NONE)
                .frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .lineWidth(1.0f);

        VkPipelineMultisampleStateCreateInfo multisample = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

        VkPipelineDepthStencilStateCreateInfo depthStencil = VkPipelineDepthStencilStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                .depthTestEnable(true)
                .depthWriteEnable(true)
                .depthCompareOp(VK_COMPARE_OP_LESS);

        VkPipelineColorBlendAttachmentState.Buffer blendAttachment =
                VkPipelineColorBlendAttachmentState.calloc(1, stack)
                        .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
                        .blendEnable(false);
        VkPipelineColorBlendStateCreateInfo colorBlend = VkPipelineColorBlendStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
                .pAttachments(blendAttachment);

        VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        if (descriptorSetLayout != VK_NULL_HANDLE) {   // set 0 = the combined image samplers
            layoutInfo.pSetLayouts(stack.longs(descriptorSetLayout));
        }
        if (options.mvp()) {   // mvp + model + camera, read by the vertex and fragment stages
            layoutInfo.pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                    .offset(0).size(PUSH_BYTES));
        }
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreatePipelineLayout(device, layoutInfo, null, pLayout), "vkCreatePipelineLayout");
        pipelineLayout = pLayout.get(0);

        VkGraphicsPipelineCreateInfo.Buffer info = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                .pStages(stages)
                .pVertexInputState(vertexInput)
                .pInputAssemblyState(inputAssembly)
                .pViewportState(viewportState)
                .pRasterizationState(rasterizer)
                .pMultisampleState(multisample)
                .pDepthStencilState(depthStencil)
                .pColorBlendState(colorBlend)
                .layout(pipelineLayout)
                .renderPass(renderPass)
                .subpass(0);
        LongBuffer pPipeline = stack.mallocLong(1);
        check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, info, null, pPipeline),
                "vkCreateGraphicsPipelines");
        graphicsPipeline = pPipeline.get(0);
    }

    private long createShaderModule(byte[] spirv, MemoryStack stack) {
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

    /** Maps a v1 attribute element {@link Type} (f32 scalar/vector) to its Vulkan vertex format. */
    private static int vertexFormat(Type type) {
        if (type instanceof Type.Float) {
            return VK_FORMAT_R32_SFLOAT;
        }
        if (type instanceof Type.Vector v) {
            return switch (v.count()) {
                case 2 -> VK_FORMAT_R32G32_SFLOAT;
                case 3 -> VK_FORMAT_R32G32B32_SFLOAT;
                case 4 -> VK_FORMAT_R32G32B32A32_SFLOAT;
                default -> throw new IllegalArgumentException("unsupported vertex vector width: " + v.count());
            };
        }
        throw new IllegalArgumentException("unsupported vertex attribute type: " + type);
    }

    private void createVertexAndIndexBuffers(Mesh mesh, MemoryStack stack) {
        indexCount = mesh.indexCount();

        long vertexBytes = (long) mesh.vertices().length * Float.BYTES;
        vertexBuffer = createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, stack);
        vertexMemory = bindHostVisible(vertexBuffer, stack);
        writeFloats(vertexMemory, mesh.vertices());

        long indexBytes = (long) mesh.indices().length * Integer.BYTES;
        indexBuffer = createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, stack);
        indexMemory = bindHostVisible(indexBuffer, stack);
        writeInts(indexMemory, mesh.indices());
    }

    private long createBuffer(long sizeBytes, int usage, MemoryStack stack) {
        VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                .size(sizeBytes)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
        LongBuffer pBuffer = stack.mallocLong(1);
        check(vkCreateBuffer(device, info, null, pBuffer), "vkCreateBuffer");
        return pBuffer.get(0);
    }

    private long bindHostVisible(long buffer, MemoryStack stack) {
        VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
        vkGetBufferMemoryRequirements(device, buffer, requirements);
        long memory = allocate(requirements,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stack);
        check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        return memory;
    }

    private long allocate(VkMemoryRequirements requirements, int properties, MemoryStack stack) {
        VkMemoryAllocateInfo info = VkMemoryAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                .allocationSize(requirements.size())
                .memoryTypeIndex(findMemoryType(requirements.memoryTypeBits(), properties, stack));
        LongBuffer pMemory = stack.mallocLong(1);
        check(vkAllocateMemory(device, info, null, pMemory), "vkAllocateMemory");
        return pMemory.get(0);
    }

    private int findMemoryType(int typeFilter, int properties, MemoryStack stack) {
        VkPhysicalDeviceMemoryProperties memProps = VkPhysicalDeviceMemoryProperties.malloc(stack);
        vkGetPhysicalDeviceMemoryProperties(physical, memProps);
        for (int i = 0; i < memProps.memoryTypeCount(); i++) {
            boolean allowed = (typeFilter & (1 << i)) != 0;
            boolean matches = (memProps.memoryTypes(i).propertyFlags() & properties) == properties;
            if (allowed && matches) {
                return i;
            }
        }
        throw new IllegalStateException("no memory type with properties 0x" + Integer.toHexString(properties));
    }

    private void writeFloats(long memory, float[] data) {
        try (MemoryStack stack = stackPush()) {
            long size = (long) data.length * Float.BYTES;
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, size, 0, pData), "vkMapMemory");
            MemoryUtil.memByteBuffer(pData.get(0), (int) size).asFloatBuffer().put(data);
            vkUnmapMemory(device, memory);
        }
    }

    private void writeInts(long memory, int[] data) {
        try (MemoryStack stack = stackPush()) {
            long size = (long) data.length * Integer.BYTES;
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, size, 0, pData), "vkMapMemory");
            MemoryUtil.memByteBuffer(pData.get(0), (int) size).asIntBuffer().put(data);
            vkUnmapMemory(device, memory);
        }
    }

    private void createTextureResources(MemoryStack stack) {
        int n = options.textures().size() + options.cubemaps().size();
        if (n == 0) {
            return;
        }
        textureBindings = new int[n];
        textureImages = new long[n];
        textureMemories = new long[n];
        textureViews = new long[n];
        textureSamplers = new long[n];

        int index = 0;
        for (Map.Entry<Integer, Path> entry : options.textures().entrySet()) {
            textureBindings[index] = entry.getKey();
            loadTexture(index, entry.getValue());
            index++;
        }
        for (Map.Entry<Integer, Path> entry : options.cubemaps().entrySet()) {
            textureBindings[index] = entry.getKey();
            loadCubemap(index, entry.getValue());
            index++;
        }
        createDescriptorSet(stack, n);
    }

    private void loadTexture(int index, Path path) {
        try (MemoryStack stack = stackPush()) {
            IntBuffer w = stack.mallocInt(1);
            IntBuffer h = stack.mallocInt(1);
            IntBuffer channels = stack.mallocInt(1);
            ByteBuffer pixels = stbi_load(path.toString(), w, h, channels, 4);   // force 4-channel RGBA
            if (pixels == null) {
                throw new IllegalStateException("failed to load texture: " + path);
            }
            int width = w.get(0);
            int height = h.get(0);
            long size = (long) width * height * 4;
            try {
                long staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stack);
                long stagingMemory = bindHostVisible(staging, stack);
                writeBytes(stagingMemory, pixels, size);

                textureImages[index] = createTextureImage(width, height, stack);
                textureMemories[index] = bindImageMemory(textureImages[index], stack);
                uploadTexture(textureImages[index], staging, width, height, stack);

                vkDestroyBuffer(device, staging, null);
                vkFreeMemory(device, stagingMemory, null);
            } finally {
                stbi_image_free(pixels);
            }
            textureViews[index] = createImageView(textureImages[index], VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT, stack);
            textureSamplers[index] = createSampler(stack);
        }
    }

    private long createTextureImage(int width, int height, MemoryStack stack) {
        VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .imageType(VK_IMAGE_TYPE_2D)
                .format(VK_FORMAT_R8G8B8A8_UNORM)
                .mipLevels(1).arrayLayers(1)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        info.extent().width(width).height(height).depth(1);
        LongBuffer pImage = stack.mallocLong(1);
        check(vkCreateImage(device, info, null, pImage), "vkCreateImage(texture)");
        return pImage.get(0);
    }

    private long bindImageMemory(long image, MemoryStack stack) {
        VkMemoryRequirements requirements = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(device, image, requirements);
        long memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, stack);
        check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory(texture)");
        return memory;
    }

    /** Transitions the image to transfer-dst, copies the staging buffer in, and transitions it to shader-read. */
    private void uploadTexture(long image, long staging, int width, int height, MemoryStack stack) {
        VkCommandBuffer cmd = beginOneTimeCommands(stack);

        transitionImage(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, stack);

        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
        region.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0)
                .baseArrayLayer(0).layerCount(1);
        region.imageOffset().set(0, 0, 0);
        region.imageExtent().width(width).height(height).depth(1);
        vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

        transitionImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 1, stack);

        endOneTimeCommands(cmd, stack);
    }

    /** Loads six face PNGs ({@code <prefix>_px/_nx/_py/_ny/_pz/_nz.png}) into a cubemap image + sampler. */
    private void loadCubemap(int index, Path prefix) {
        String[] suffixes = {"_px", "_nx", "_py", "_ny", "_pz", "_nz"};
        try (MemoryStack stack = stackPush()) {
            ByteBuffer[] faces = new ByteBuffer[6];
            IntBuffer w = stack.mallocInt(1);
            IntBuffer h = stack.mallocInt(1);
            IntBuffer channels = stack.mallocInt(1);
            int width = 0;
            int height = 0;
            try {
                for (int f = 0; f < 6; f++) {
                    Path facePath = Path.of(prefix + suffixes[f] + ".png");
                    faces[f] = stbi_load(facePath.toString(), w, h, channels, 4);
                    if (faces[f] == null) {
                        throw new IllegalStateException("failed to load cubemap face: " + facePath);
                    }
                    if (f == 0) {
                        width = w.get(0);
                        height = h.get(0);
                    }
                }
                long faceSize = (long) width * height * 4;
                long staging = createBuffer(faceSize * 6, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stack);
                long stagingMemory = bindHostVisible(staging, stack);
                writeFaces(stagingMemory, faces, faceSize);

                textureImages[index] = createCubeImage(width, height, stack);
                textureMemories[index] = bindImageMemory(textureImages[index], stack);
                uploadCubemap(textureImages[index], staging, width, height, faceSize, stack);

                vkDestroyBuffer(device, staging, null);
                vkFreeMemory(device, stagingMemory, null);
            } finally {
                for (ByteBuffer face : faces) {
                    if (face != null) {
                        stbi_image_free(face);
                    }
                }
            }
            textureViews[index] = createCubeView(textureImages[index], stack);
            textureSamplers[index] = createSampler(stack);
        }
    }

    private long createCubeImage(int width, int height, MemoryStack stack) {
        VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
                .flags(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
                .imageType(VK_IMAGE_TYPE_2D)
                .format(VK_FORMAT_R8G8B8A8_UNORM)
                .mipLevels(1).arrayLayers(6)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        info.extent().width(width).height(height).depth(1);
        LongBuffer pImage = stack.mallocLong(1);
        check(vkCreateImage(device, info, null, pImage), "vkCreateImage(cubemap)");
        return pImage.get(0);
    }

    private long createCubeView(long image, MemoryStack stack) {
        VkImageViewCreateInfo info = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(image)
                .viewType(VK_IMAGE_VIEW_TYPE_CUBE)
                .format(VK_FORMAT_R8G8B8A8_UNORM);
        info.subresourceRange()
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(6);
        LongBuffer pView = stack.mallocLong(1);
        check(vkCreateImageView(device, info, null, pView), "vkCreateImageView(cubemap)");
        return pView.get(0);
    }

    /** Copies the six staged faces into the cube image's array layers (Vulkan order +X,-X,+Y,-Y,+Z,-Z). */
    private void uploadCubemap(long image, long staging, int width, int height, long faceSize, MemoryStack stack) {
        VkCommandBuffer cmd = beginOneTimeCommands(stack);

        transitionImage(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 6, stack);

        VkBufferImageCopy.Buffer regions = VkBufferImageCopy.calloc(6, stack);
        for (int f = 0; f < 6; f++) {
            regions.get(f).bufferOffset(faceSize * f).bufferRowLength(0).bufferImageHeight(0);
            regions.get(f).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0)
                    .baseArrayLayer(f).layerCount(1);
            regions.get(f).imageOffset().set(0, 0, 0);
            regions.get(f).imageExtent().width(width).height(height).depth(1);
        }
        vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions);

        transitionImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 6, stack);

        endOneTimeCommands(cmd, stack);
    }

    private void writeFaces(long memory, ByteBuffer[] faces, long faceSize) {
        try (MemoryStack stack = stackPush()) {
            long total = faceSize * faces.length;
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, total, 0, pData), "vkMapMemory");
            long dst = pData.get(0);
            for (int f = 0; f < faces.length; f++) {
                MemoryUtil.memCopy(MemoryUtil.memAddress(faces[f]), dst + faceSize * f, faceSize);
            }
            vkUnmapMemory(device, memory);
        }
    }

    private void transitionImage(VkCommandBuffer cmd, long image, int oldLayout, int newLayout,
            int srcAccess, int dstAccess, int srcStage, int dstStage, int layerCount, MemoryStack stack) {
        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .oldLayout(oldLayout).newLayout(newLayout)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(image).srcAccessMask(srcAccess).dstAccessMask(dstAccess);
        barrier.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(layerCount);
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, barrier);
    }

    private VkCommandBuffer beginOneTimeCommands(MemoryStack stack) {
        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
        PointerBuffer pCmd = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, allocInfo, pCmd), "vkAllocateCommandBuffers");
        VkCommandBuffer cmd = new VkCommandBuffer(pCmd.get(0), device);
        VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        check(vkBeginCommandBuffer(cmd, begin), "vkBeginCommandBuffer");
        return cmd;
    }

    private void endOneTimeCommands(VkCommandBuffer cmd, MemoryStack stack) {
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO).pCommandBuffers(stack.pointers(cmd));
        check(vkQueueSubmit(queue, submit, VK_NULL_HANDLE), "vkQueueSubmit(oneTime)");
        check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        vkFreeCommandBuffers(device, commandPool, cmd);
    }

    private long createSampler(MemoryStack stack) {
        VkSamplerCreateInfo info = VkSamplerCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO)
                .magFilter(VK_FILTER_LINEAR).minFilter(VK_FILTER_LINEAR)
                .addressModeU(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                .addressModeV(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                .addressModeW(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                .minLod(0).maxLod(0).maxAnisotropy(1.0f);
        LongBuffer pSampler = stack.mallocLong(1);
        check(vkCreateSampler(device, info, null, pSampler), "vkCreateSampler");
        return pSampler.get(0);
    }

    private void createDescriptorSet(MemoryStack stack, int n) {
        VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(n, stack);
        for (int i = 0; i < n; i++) {
            bindings.get(i).binding(textureBindings[i])
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_FRAGMENT_BIT);
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO).pBindings(bindings);
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreateDescriptorSetLayout(device, layoutInfo, null, pLayout), "vkCreateDescriptorSetLayout");
        descriptorSetLayout = pLayout.get(0);

        VkDescriptorPoolSize.Buffer poolSize = VkDescriptorPoolSize.calloc(1, stack)
                .type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(n);
        VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO).maxSets(1).pPoolSizes(poolSize);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateDescriptorPool(device, poolInfo, null, pPool), "vkCreateDescriptorPool");
        descriptorPool = pPool.get(0);

        VkDescriptorSetAllocateInfo allocInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                .descriptorPool(descriptorPool).pSetLayouts(stack.longs(descriptorSetLayout));
        LongBuffer pSet = stack.mallocLong(1);
        check(vkAllocateDescriptorSets(device, allocInfo, pSet), "vkAllocateDescriptorSets");
        descriptorSet = pSet.get(0);

        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(n, stack);
        for (int i = 0; i < n; i++) {
            VkDescriptorImageInfo.Buffer imageInfo = VkDescriptorImageInfo.calloc(1, stack)
                    .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    .imageView(textureViews[i]).sampler(textureSamplers[i]);
            writes.get(i).sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                    .dstSet(descriptorSet).dstBinding(textureBindings[i]).dstArrayElement(0)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .pImageInfo(imageInfo);
        }
        vkUpdateDescriptorSets(device, writes, null);
    }

    private void writeBytes(long memory, ByteBuffer data, long size) {
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, size, 0, pData), "vkMapMemory");
            MemoryUtil.memCopy(MemoryUtil.memAddress(data), pData.get(0), size);
            vkUnmapMemory(device, memory);
        }
    }

    private void createCommandPoolAndBuffer(MemoryStack stack) {
        VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(queueFamily);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, poolInfo, null, pPool), "vkCreateCommandPool");
        commandPool = pPool.get(0);

        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(1);
        PointerBuffer pBuffer = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, allocInfo, pBuffer), "vkAllocateCommandBuffers");
        commandBuffer = new VkCommandBuffer(pBuffer.get(0), device);
    }

    /** Records the draw for {@code framebuffer}: clear, bind pipeline + buffers, indexed draw. */
    private void recordDraw(long framebuffer, MemoryStack stack) {
        VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        check(vkBeginCommandBuffer(commandBuffer, begin), "vkBeginCommandBuffer");

        VkClearValue.Buffer clears = VkClearValue.calloc(2, stack);
        clears.get(0).color().float32(0, CLEAR_RGBA[0]).float32(1, CLEAR_RGBA[1])
                .float32(2, CLEAR_RGBA[2]).float32(3, CLEAR_RGBA[3]);
        clears.get(1).depthStencil().depth(1.0f).stencil(0);

        VkRect2D area = VkRect2D.calloc(stack);
        area.offset().set(0, 0);
        area.extent().width(extentWidth).height(extentHeight);

        VkRenderPassBeginInfo rpBegin = VkRenderPassBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
                .renderPass(renderPass)
                .framebuffer(framebuffer)
                .renderArea(area)
                .pClearValues(clears);
        vkCmdBeginRenderPass(commandBuffer, rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        if (descriptorSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                    stack.longs(descriptorSet), null);
        }
        if (options.mvp()) {
            vkCmdPushConstants(commandBuffer, pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushConstantData(stack));
        }
        vkCmdBindVertexBuffers(commandBuffer, 0, stack.longs(vertexBuffer), stack.longs(0));
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    /** The per-frame push-constant block: {@code mvp} (0), {@code model} (64), {@code cameraPosition} (128). */
    private ByteBuffer pushConstantData(MemoryStack stack) {
        float aspect = (float) extentWidth / extentHeight;
        float angle = (float) glfwGetTime() * 0.6f + 0.6f;   // spin; base offset so a one-frame shot isn't flat-on
        Matrix4f model = new Matrix4f().rotateY(angle).rotateX(0.35f);
        Matrix4f viewProjection = new Matrix4f()
                .perspective((float) Math.toRadians(45.0), aspect, 0.1f, 20.0f, true)   // Vulkan depth [0,1]
                .lookAt(CAMERA_EYE[0], CAMERA_EYE[1], CAMERA_EYE[2], 0, 0, 0, 0, 1, 0);
        Matrix4f mvp = new Matrix4f(viewProjection).mul(model);

        ByteBuffer buffer = stack.malloc(PUSH_BYTES);
        mvp.get(0, buffer);                              // column-major, matching the shader's ColMajor mat4
        model.get(64, buffer);
        buffer.putFloat(128, CAMERA_EYE[0]).putFloat(132, CAMERA_EYE[1]).putFloat(136, CAMERA_EYE[2]);
        return buffer;
    }

    private void createSyncObjects(MemoryStack stack) {
        VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                .flags(VK_FENCE_CREATE_SIGNALED_BIT);   // start signaled so the first wait returns immediately
        LongBuffer p = stack.mallocLong(1);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        imageAvailable = p.get(0);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        renderFinished = p.get(0);
        check(vkCreateFence(device, fenceInfo, null, p), "vkCreateFence");
        inFlight = p.get(0);
    }

    private void loop() {
        int rendered = 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            rendered++;
            if (options.frames().isPresent() && rendered >= options.frames().get()) {
                break;
            }
        }
        vkDeviceWaitIdle(device);
        System.out.println("[preview] rendered " + rendered + " frame(s); " + indexCount + " indices");
    }

    private void drawFrame() {
        try (MemoryStack stack = stackPush()) {
            check(vkWaitForFences(device, stack.longs(inFlight), true, Long.MAX_VALUE), "vkWaitForFences");
            check(vkResetFences(device, stack.longs(inFlight)), "vkResetFences");

            IntBuffer pImageIndex = stack.mallocInt(1);
            check(vkAcquireNextImageKHR(device, swapchain, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE,
                    pImageIndex), "vkAcquireNextImageKHR");
            int imageIndex = pImageIndex.get(0);

            check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
            recordDraw(framebuffers[imageIndex], stack);

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .waitSemaphoreCount(1)
                    .pWaitSemaphores(stack.longs(imageAvailable))
                    .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT))
                    .pCommandBuffers(stack.pointers(commandBuffer))
                    .pSignalSemaphores(stack.longs(renderFinished));
            check(vkQueueSubmit(queue, submit, inFlight), "vkQueueSubmit");

            VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
                    .pWaitSemaphores(stack.longs(renderFinished))
                    .swapchainCount(1)
                    .pSwapchains(stack.longs(swapchain))
                    .pImageIndices(stack.ints(imageIndex));
            check(vkQueuePresentKHR(queue, present), "vkQueuePresentKHR");
        }
    }

    /** Renders one frame off-screen, copies the rendered image to host memory, and writes it as a PNG. */
    private void captureScreenshot(Path output) {
        try (MemoryStack stack = stackPush()) {
            check(vkWaitForFences(device, stack.longs(inFlight), true, Long.MAX_VALUE), "vkWaitForFences");
            check(vkResetFences(device, stack.longs(inFlight)), "vkResetFences");

            IntBuffer pImageIndex = stack.mallocInt(1);
            check(vkAcquireNextImageKHR(device, swapchain, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE,
                    pImageIndex), "vkAcquireNextImageKHR");
            int imageIndex = pImageIndex.get(0);

            check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
            recordDraw(framebuffers[imageIndex], stack);

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .waitSemaphoreCount(1)
                    .pWaitSemaphores(stack.longs(imageAvailable))
                    .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT))
                    .pCommandBuffers(stack.pointers(commandBuffer));
            check(vkQueueSubmit(queue, submit, inFlight), "vkQueueSubmit");
            check(vkWaitForFences(device, stack.longs(inFlight), true, Long.MAX_VALUE), "vkWaitForFences");

            long bytes = (long) extentWidth * extentHeight * 4;
            long buffer = createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, stack);
            long memory = bindHostVisible(buffer, stack);
            copyImageToBuffer(swapchainImages[imageIndex], buffer, stack);
            writePng(output, memory, bytes);
            vkFreeMemory(device, memory, null);
            vkDestroyBuffer(device, buffer, null);
        }
        System.out.println("[preview] wrote " + extentWidth + "x" + extentHeight + " screenshot to " + output);
    }

    /** Transitions the presented image to a transfer source and copies it into {@code buffer}. */
    private void copyImageToBuffer(long image, long buffer, MemoryStack stack) {
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

        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER)
                .oldLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
                .newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(image)
                .srcAccessMask(0)
                .dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);
        barrier.subresourceRange()
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, null, null, barrier);

        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack)
                .bufferOffset(0).bufferRowLength(0).bufferImageHeight(0);
        region.imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0)
                .baseArrayLayer(0).layerCount(1);
        region.imageOffset().set(0, 0, 0);
        region.imageExtent().width(extentWidth).height(extentHeight).depth(1);
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, region);

        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                .pCommandBuffers(stack.pointers(cmd));
        check(vkQueueSubmit(queue, submit, VK_NULL_HANDLE), "vkQueueSubmit(copy)");
        check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        vkFreeCommandBuffers(device, commandPool, cmd);
    }

    /** Reads RGBA pixels from mapped {@code memory} (swizzling BGRA swapchains) and writes a PNG. */
    private void writePng(Path output, long memory, long bytes) {
        ensureParent(output);
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, memory, 0, bytes, 0, pData), "vkMapMemory");
            ByteBuffer src = MemoryUtil.memByteBuffer(pData.get(0), (int) bytes);
            boolean bgra = swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM
                    || swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB;

            ByteBuffer pixels = MemoryUtil.memAlloc((int) bytes);
            for (int i = 0; i < extentWidth * extentHeight; i++) {
                int k = i * 4;
                byte c0 = src.get(k);
                byte c1 = src.get(k + 1);
                byte c2 = src.get(k + 2);
                byte c3 = src.get(k + 3);
                if (bgra) {
                    pixels.put(c2).put(c1).put(c0).put(c3);   // BGRA → RGBA
                } else {
                    pixels.put(c0).put(c1).put(c2).put(c3);
                }
            }
            pixels.flip();
            vkUnmapMemory(device, memory);

            boolean ok = stbi_write_png(output.toString(), extentWidth, extentHeight, 4, pixels, extentWidth * 4);
            MemoryUtil.memFree(pixels);
            if (!ok) {
                throw new IllegalStateException("stbi_write_png failed for " + output);
            }
        }
    }

    private static void ensureParent(Path output) {
        Path parent = output.getParent();
        if (parent == null) {
            return;
        }
        try {
            Files.createDirectories(parent);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot create screenshot directory " + parent, e);
        }
    }

    @Override
    public void close() {
        if (device != null) {
            vkDeviceWaitIdle(device);
            destroyIf(inFlight, h -> vkDestroyFence(device, h, null));
            destroyIf(renderFinished, h -> vkDestroySemaphore(device, h, null));
            destroyIf(imageAvailable, h -> vkDestroySemaphore(device, h, null));
            destroyIf(commandPool, h -> vkDestroyCommandPool(device, h, null));
            destroyIf(indexBuffer, h -> vkDestroyBuffer(device, h, null));
            destroyIf(indexMemory, h -> vkFreeMemory(device, h, null));
            destroyIf(vertexBuffer, h -> vkDestroyBuffer(device, h, null));
            destroyIf(vertexMemory, h -> vkFreeMemory(device, h, null));
            destroyIf(graphicsPipeline, h -> vkDestroyPipeline(device, h, null));
            destroyIf(pipelineLayout, h -> vkDestroyPipelineLayout(device, h, null));
            destroyIf(fragmentShaderModule, h -> vkDestroyShaderModule(device, h, null));
            destroyIf(vertexShaderModule, h -> vkDestroyShaderModule(device, h, null));
            for (long sampler : textureSamplers) {
                destroyIf(sampler, h -> vkDestroySampler(device, h, null));
            }
            for (long view : textureViews) {
                destroyIf(view, h -> vkDestroyImageView(device, h, null));
            }
            for (long image : textureImages) {
                destroyIf(image, h -> vkDestroyImage(device, h, null));
            }
            for (long memory : textureMemories) {
                destroyIf(memory, h -> vkFreeMemory(device, h, null));
            }
            destroyIf(descriptorPool, h -> vkDestroyDescriptorPool(device, h, null));
            destroyIf(descriptorSetLayout, h -> vkDestroyDescriptorSetLayout(device, h, null));
            for (long framebuffer : framebuffers) {
                vkDestroyFramebuffer(device, framebuffer, null);
            }
            destroyIf(depthView, h -> vkDestroyImageView(device, h, null));
            destroyIf(depthImage, h -> vkDestroyImage(device, h, null));
            destroyIf(depthMemory, h -> vkFreeMemory(device, h, null));
            destroyIf(renderPass, h -> vkDestroyRenderPass(device, h, null));
            for (long view : imageViews) {
                vkDestroyImageView(device, view, null);
            }
            destroyIf(swapchain, h -> vkDestroySwapchainKHR(device, h, null));
            vkDestroyDevice(device, null);
        }
        if (surface != VK_NULL_HANDLE && instance != null) {
            vkDestroySurfaceKHR(instance, surface, null);
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
        }
        if (window != NULL) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

    private interface Destroyer {
        void destroy(long handle);
    }

    private static void destroyIf(long handle, Destroyer destroyer) {
        if (handle != VK_NULL_HANDLE) {
            destroyer.destroy(handle);
        }
    }

    private static byte[] readBytes(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + path, e);
        }
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
